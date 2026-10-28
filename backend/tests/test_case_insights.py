"""
Tests for the case-insights endpoints (signal scorecard, policy trace,
counterfactuals, case activity timeline) and the document dossier
(side-by-side invoice comparison).

Two levels of coverage:

  * fast tests that do not need the graph/RAG stores -- the scorecard
    arithmetic and the document dossier are checked directly against the
    real OCR findings on disk;
  * integration tests (skipped when the shared graph/RAG stores have not
    been built) that run the real 5-agent pipeline and then assert the
    scorecard adds back up to the case's own risk_score.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_DB_PATH = REPO_ROOT / "data" / "graph_db"
CHROMA_PATH = REPO_ROOT / "data" / "chroma_db"

integration = pytest.mark.skipif(
    not GRAPH_DB_PATH.exists() or not CHROMA_PATH.exists(),
    reason="run scripts/ingest_graph.py and scripts/evaluate_rag.py first",
)


# ---------------------------------------------------------------------------
# Signal scorecard
# ---------------------------------------------------------------------------

class TestSignalScorecard:
    def test_scorecard_lists_every_signal_even_when_unavailable(self, client):
        created = client.post("/api/cases", json={
            "source": "manual", "domain": "financial_transaction",
            "amount": 100.0, "data_source": "user_submitted",
        }).json()

        body = client.get(f"/api/cases/{created['case_id']}/insights").json()
        names = [s["name"] for s in body["signals"]]
        assert names == ["tabular_ml_risk", "graph_risk", "temporal_risk", "document_risk"]
        assert body["investigated"] is False
        # nothing is claimed for a case that has not been investigated
        assert all(s["available"] is False for s in body["signals"])
        assert all(s["contribution"] is None for s in body["signals"])
        # and every unavailable signal explains itself
        assert all(s["note"] for s in body["signals"])

    def test_scorecard_contributions_sum_to_the_fused_score(self, client):
        """The parts must add back up to the whole -- otherwise the
        scorecard is decoration rather than a decomposition."""
        from app.ml.fusion.risk_fusion import fuse

        signals = {"tabular_ml_risk": None, "graph_risk": 0.8,
                   "temporal_risk": 0.4, "document_risk": 0.9}
        expected = fuse(signals)["risk_score"]
        total_weight = 0.30 + 0.15 + 0.20

        contribution = (0.8 * 0.30 + 0.4 * 0.15 + 0.9 * 0.20) / total_weight
        # fuse() rounds the score to 4 decimals, so the two agree to within
        # half a rounding step.
        assert abs(contribution - expected) < 1e-4

    def test_insights_404s_for_unknown_case(self, client):
        assert client.get("/api/cases/FS-NOPE/insights").status_code == 404


# ---------------------------------------------------------------------------
# Document dossier
# ---------------------------------------------------------------------------

class TestDocumentDossier:
    def test_case_without_document_reports_no_document(self, client):
        created = client.post("/api/cases", json={
            "source": "manual", "domain": "insurance_claim",
            "amount": 100.0, "data_source": "user_submitted",
        }).json()

        body = client.get(f"/api/cases/{created['case_id']}/documents").json()
        assert body["has_document"] is False
        assert body["documents"] == []

    def test_documents_endpoint_404s_for_unknown_case(self, client):
        assert client.get("/api/cases/FS-NOPE/documents").status_code == 404

    def test_duplicate_invoice_group_is_returned_side_by_side(self, client):
        """FS-SYN-00019 shares invoice INV-81793 with two other cases."""
        client.post("/api/cases/seed-synthetic")
        body = client.get("/api/cases/FS-SYN-00019/documents").json()
        assert body["has_document"] is True
        assert body["duplicate_across_cases"] is True
        assert body["group_invoice_number"] == "INV-81793"
        assert len(body["documents"]) == 3
        assert {d["case_id"] for d in body["documents"]} == {
            "FS-SYN-00018", "FS-SYN-00019", "FS-SYN-00020"}

    def test_comparison_rows_flag_the_odd_value_out(self, client):
        """Three cases filed the SAME invoice under three DIFFERENT
        claimant names -- the claimant row is the discrepancy."""
        client.post("/api/cases/seed-synthetic")
        body = client.get("/api/cases/FS-SYN-00019/documents").json()
        claimant_row = next(r for r in body["rows"] if r["label"] == "Claimant name")
        assert claimant_row["differs"] is True
        assert any(cell["highlight"] for cell in claimant_row["cells"])

        invoice_row = next(r for r in body["rows"] if r["label"] == "Invoice number")
        assert invoice_row["differs"] is False

    def test_amount_mismatch_is_highlighted(self, client):
        """FS-SYN-00017 has $7,500 printed on the invoice but $5,000 on file."""
        client.post("/api/cases/seed-synthetic")
        body = client.get("/api/cases/FS-SYN-00017/documents").json()
        amount_row = next(r for r in body["rows"] if r["label"] == "Amount printed")
        assert any(cell["highlight"] for cell in amount_row["cells"])
        assert "claim_amount_mismatch" in body["documents"][0]["flags"]

    def test_document_image_is_served_for_a_known_file(self, client):
        client.post("/api/cases/seed-synthetic")
        body = client.get("/api/cases/FS-SYN-00017/documents").json()
        image_name = body["documents"][0]["image_name"]
        r = client.get(f"/api/documents/{image_name}")
        assert r.status_code == 200
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG bytes

    def test_unknown_document_image_is_404(self, client):
        assert client.get("/api/documents/not-a-real-document.png").status_code == 404

    def test_document_image_rejects_path_traversal(self, client):
        """Serving is whitelisted against the manifest, so no traversal
        surface exists at all."""
        for attempt in ["..%2F..%2F.env", "../../.env", "..%5C..%5C.env"]:
            assert client.get(f"/api/documents/{attempt}").status_code == 404


# ---------------------------------------------------------------------------
# Full-pipeline integration
# ---------------------------------------------------------------------------

@integration
class TestInsightsAfterRealInvestigation:
    def test_scorecard_matches_the_case_risk_score(self, client):
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        ring_case = next(c for c in cases if c["scenario_id"] == "FS-SC-003")

        client.post(f"/api/cases/{ring_case['case_id']}/investigate")
        detail = client.get(f"/api/cases/{ring_case['case_id']}").json()
        insights = client.get(f"/api/cases/{ring_case['case_id']}/insights").json()

        assert insights["investigated"] is True
        assert len(insights["agents"]) == 5

        available = [s for s in insights["signals"] if s["available"]]
        assert available, "the ring case must have at least one real signal"
        total = sum(s["contribution"] for s in available)
        assert abs(total - insights["risk_score"]) < 0.01
        assert abs(insights["risk_score"] - detail["case"]["risk_score"]) < 1e-6

    def test_policy_trace_records_the_conditions_that_fired(self, client):
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        ring_case = next(c for c in cases if c["scenario_id"] == "FS-SC-003")

        client.post(f"/api/cases/{ring_case['case_id']}/investigate")
        insights = client.get(f"/api/cases/{ring_case['case_id']}/insights").json()

        policy = insights["policy"]
        assert policy["decision"] == "ESCALATE"
        assert policy["conditions_triggered"]
        assert policy["policy_version"]
        # the ring override is a policy override, so the routing was not
        # score-driven -- the scorecard must say so
        assert policy["score_driven"] is False
        assert policy["overrides"]

    def test_counterfactuals_are_arithmetically_correct(self, client):
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        ring_case = next(c for c in cases if c["scenario_id"] == "FS-SC-003")

        client.post(f"/api/cases/{ring_case['case_id']}/investigate")
        insights = client.get(f"/api/cases/{ring_case['case_id']}/insights").json()

        counterfactuals = insights["counterfactuals"]
        assert counterfactuals
        for c in counterfactuals:
            assert c["explanation"]
            if c["achievable"]:
                assert c["required_value"] is not None
                assert c["required_value"] < c["current_value"]
            # an active override must be disclosed, never hidden
            if insights["policy"]["overrides"]:
                assert "override" in c["explanation"]

    def test_timeline_lists_the_cases_events(self, client):
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        ring_case = next(c for c in cases if c["scenario_id"] == "FS-SC-003")

        client.post(f"/api/cases/{ring_case['case_id']}/investigate")
        insights = client.get(f"/api/cases/{ring_case['case_id']}/insights").json()

        events = insights["events"]
        assert events
        assert any(e["is_current"] for e in events)
        # events are returned in chronological order
        stamps = [e["timestamp"] for e in events if e["timestamp"]]
        assert stamps == sorted(stamps)

    def test_velocity_burst_shows_up_as_a_cluster(self, client):
        """Scenario FS-SC-002 files a burst of claims in a tight window."""
        client.post("/api/cases/seed-synthetic")
        cases = client.get("/api/cases?limit=100").json()
        burst = next((c for c in cases if c["scenario_id"] == "FS-SC-002"), None)
        if burst is None:
            pytest.skip("scenario FS-SC-002 not present in the scenario library")

        client.post(f"/api/cases/{burst['case_id']}/investigate")
        insights = client.get(f"/api/cases/{burst['case_id']}/insights").json()
        assert insights["velocity_cluster_size"] >= 1
        assert insights["velocity_window_seconds"] > 0
