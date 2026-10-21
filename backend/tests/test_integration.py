"""
Integration tests -- exercises the FULL stack through the actual FastAPI
TestClient: seed -> investigate (real 5-agent pipeline, real graph/RAG/
document tool calls) -> persisted DB rows -> real PDF download -> real
analytics. This is the test suite equivalent of the live curl-based
verification run during development (see docs/EXPERIMENTS.md), so a
future change that breaks the integration gets caught automatically.

Requires the graph store (kuzu) and RAG index (ChromaDB) to already be
built (scripts/ingest_graph.py, scripts/evaluate_rag.py) -- these are
shared, persistent stores outside the per-test SQLite database, so tests
here skip gracefully if they haven't been built yet rather than failing
with a confusing error.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_DB_PATH = REPO_ROOT / "data" / "graph_db"
CHROMA_PATH = REPO_ROOT / "data" / "chroma_db"

pytestmark = pytest.mark.skipif(
    not GRAPH_DB_PATH.exists() or not CHROMA_PATH.exists(),
    reason="run scripts/ingest_graph.py and scripts/evaluate_rag.py first",
)


class TestFullIntegrationFlow:
    def test_seed_synthetic_creates_30_cases(self, client):
        r = client.post("/api/cases/seed-synthetic")
        assert r.status_code == 200
        assert r.json()["seeded"] == 30

        cases = client.get("/api/cases?limit=100").json()
        assert len(cases) == 30

    def test_seed_is_idempotent(self, client):
        client.post("/api/cases/seed-synthetic")
        r2 = client.post("/api/cases/seed-synthetic")
        assert r2.json()["seeded"] == 0
        assert r2.json()["already_existed"] == 30

    def test_investigate_fraud_ring_case_escalates(self, client):
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        ring_case = next(c for c in cases if c["scenario_id"] == "FS-SC-003")

        r = client.post(f"/api/cases/{ring_case['case_id']}/investigate")
        assert r.status_code == 200
        result = r.json()
        assert result["routing_decision"]["decision"] == "ESCALATE"
        assert result["n_agents_run"] == 5
        assert result["n_evidence_items"] > 0

    def test_manual_case_with_no_signals_is_not_auto_closed(self, client):
        created = client.post("/api/cases", json={
            "source": "manual",
            "domain": "financial_transaction",
            "amount": 125.00,
            "data_source": "user_submitted",
        })
        assert created.status_code == 200

        result = client.post(f"/api/cases/{created.json()['case_id']}/investigate")
        assert result.status_code == 200
        body = result.json()
        assert body["risk_level"] == "UNSCORED"
        assert body["routing_decision"]["decision"] == "HUMAN_REVIEW"
        assert body["n_agents_run"] == 5
        assert body["report_id"] is not None

    def test_investigate_persists_all_related_rows(self, client):
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        ring_case = next(c for c in cases if c["scenario_id"] == "FS-SC-003")

        client.post(f"/api/cases/{ring_case['case_id']}/investigate")
        detail = client.get(f"/api/cases/{ring_case['case_id']}").json()

        assert len(detail["agent_executions"]) == 5
        assert len(detail["evidence"]) > 0
        assert len(detail["reports"]) == 1
        assert len(detail["decisions"]) == 1
        assert detail["case"]["risk_level"] == "HIGH"

    def test_investigate_is_idempotent_not_a_pk_violation(self, client):
        """Regression guard: re-investigating a case must not crash with a
        primary-key violation on Decision/Report (their IDs are
        case-derived, see pipeline_service.py)."""
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        ring_case = next(c for c in cases if c["scenario_id"] == "FS-SC-003")

        r1 = client.post(f"/api/cases/{ring_case['case_id']}/investigate")
        r2 = client.post(f"/api/cases/{ring_case['case_id']}/investigate")
        assert r1.status_code == 200
        assert r2.status_code == 200

        detail = client.get(f"/api/cases/{ring_case['case_id']}").json()
        assert len(detail["decisions"]) == 1  # not 2 -- old row replaced, not duplicated

    def test_fraud_graph_endpoint_returns_real_subgraph(self, client):
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        ring_case = next(c for c in cases if c["scenario_id"] == "FS-SC-003")

        graph = client.get(f"/api/cases/{ring_case['case_id']}/graph").json()
        assert len(graph["nodes"]) > 1
        assert len(graph["edges"]) > 0

    def test_pdf_download_returns_real_pdf(self, client):
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        ring_case = next(c for c in cases if c["scenario_id"] == "FS-SC-003")
        client.post(f"/api/cases/{ring_case['case_id']}/investigate")

        r = client.get(f"/api/cases/{ring_case['case_id']}/report/pdf")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert r.content[:4] == b"%PDF"  # real PDF magic bytes, not a stub

    def test_pdf_download_404s_before_investigation(self, client):
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        uninvestigated = cases[0]
        r = client.get(f"/api/cases/{uninvestigated['case_id']}/report/pdf")
        assert r.status_code == 404

    def test_analytics_reflects_real_investigation(self, client):
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        ring_case = next(c for c in cases if c["scenario_id"] == "FS-SC-003")
        client.post(f"/api/cases/{ring_case['case_id']}/investigate")

        analytics = client.get("/api/analytics").json()
        assert analytics["total_cases"] == 30
        assert analytics["escalated"] == 1
        assert analytics["unscored"] == 29

    def test_knowledge_base_endpoint_returns_real_documents(self, client):
        r = client.get("/api/knowledge")
        assert r.status_code == 200
        data = r.json()
        assert len(data["documents"]) == 5
        assert data["stats"]["n_chunks"] > 0

    def test_knowledge_base_query_returns_relevant_results(self, client):
        r = client.post("/api/knowledge/query?question=SAR+filing+deadline&top_k=1")
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 1
        assert results[0]["document_id"] == "KB-001"

    def test_investigate_nonexistent_case_404s(self, client):
        r = client.post("/api/cases/FS-DOES-NOT-EXIST/investigate")
        assert r.status_code == 404
