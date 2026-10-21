"""Phase 7 tests -- knowledge graph store.

Kuzu is the default embedded backend. The Neo4j adapter is exercised
against a live server in the separate database smoke test.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

METRICS_PATH = REPO_ROOT / "docs" / "graph_metrics.json"


@pytest.fixture
def store(tmp_path):
    from app.graph.store import get_graph_store
    s = get_graph_store("kuzu", db_path=tmp_path / "test_graph")
    s.initialize_schema(reset=False)
    yield s
    s.close()


class TestGraphStoreBasics:
    def test_backend_selection_defaults_to_kuzu(self):
        from app.graph.store import get_graph_store
        s = get_graph_store()
        assert s.name == "kuzu"
        s.close()

    def test_unknown_backend_raises(self):
        from app.graph.store import get_graph_store
        with pytest.raises(ValueError):
            get_graph_store("mongodb")

    def test_add_and_query_customer(self, store):
        store.add_customer("C1", "Alice", data_source="synthetic", scenario_id="FS-SC-001")
        rows = store.query("MATCH (c:Customer) WHERE c.id = $cid RETURN c.name AS name", {"cid": "C1"})
        assert rows[0]["name"] == "Alice"

    def test_provenance_is_stored_on_nodes(self, store):
        """Section 8: synthetic data must be distinguishable in every store."""
        store.add_customer("C1", "Alice", data_source="synthetic", scenario_id="FS-SC-003")
        rows = store.query(
            "MATCH (c:Customer) WHERE c.id=$cid RETURN c.data_source AS ds, c.scenario_id AS sid",
            {"cid": "C1"})
        assert rows[0]["ds"] == "synthetic"
        assert rows[0]["sid"] == "FS-SC-003"

    def test_merge_is_idempotent(self, store):
        """Re-ingesting the same data must not duplicate nodes."""
        for _ in range(3):
            store.add_customer("C1", "Alice")
            store.add_entity("Device", "D1")
            store.link("USES_DEVICE", "Customer", "C1", "Device", "D1")
        count = store.query("MATCH (c:Customer) RETURN COUNT(c) AS n")[0]["n"]
        assert count == 1


class TestFraudRingQueries:
    def test_finds_customers_sharing_a_device(self, store):
        """The core 2-hop Cypher pattern: (a)-[:USES]->(x)<-[:USES]-(b)"""
        for cid, name in [("C1", "Alice"), ("C2", "Bob"), ("C3", "Carol")]:
            store.add_customer(cid, name)
        store.add_entity("Device", "SHARED-DEV")
        for cid in ["C1", "C2", "C3"]:
            store.link("USES_DEVICE", "Customer", cid, "Device", "SHARED-DEV")

        connected = store.find_connected_customers("C1")
        connected_ids = {c["connected_customer"] for c in connected}
        assert connected_ids == {"C2", "C3"}

    def test_unconnected_customer_returns_nothing(self, store):
        """Guards against a query that returns everyone regardless."""
        store.add_customer("C1", "Alice")
        store.add_customer("C2", "Bob")
        store.add_entity("Device", "D1")
        store.add_entity("Device", "D2")
        store.link("USES_DEVICE", "Customer", "C1", "Device", "D1")
        store.link("USES_DEVICE", "Customer", "C2", "Device", "D2")

        assert store.find_connected_customers("C1") == []

    def test_shared_entity_cluster_detection(self, store):
        for i in range(5):
            store.add_customer(f"C{i}", f"Person{i}")
        store.add_entity("Account", "SHARED-ACC")
        for i in range(5):
            store.link("OWNS_ACCOUNT", "Customer", f"C{i}", "Account", "SHARED-ACC")

        clusters = store.find_shared_entity_clusters("Account", min_customers=2)
        assert len(clusters) == 1
        assert clusters[0]["n_customers"] == 5

    def test_broker_excluded_from_strong_signal(self):
        """
        Design decision under test: sharing a BROKER is weak signal (many
        legitimate claimants use the same broker), so it must not appear
        in STRONG_SHARED_TYPES. Treating it as strong caused a flood of
        false positives in an earlier iteration of this project.
        """
        from app.graph.store import STRONG_SHARED_TYPES, WEAK_SHARED_TYPES
        assert "Broker" not in STRONG_SHARED_TYPES
        assert "Broker" in WEAK_SHARED_TYPES
        assert "Email" in WEAK_SHARED_TYPES

    def test_subgraph_includes_customer_and_neighbours(self, store):
        store.add_customer("C1", "Alice")
        store.add_customer("C2", "Bob")
        store.add_entity("Device", "D1")
        store.link("USES_DEVICE", "Customer", "C1", "Device", "D1")
        store.link("USES_DEVICE", "Customer", "C2", "Device", "D1")

        sub = store.get_customer_subgraph("C1")
        node_ids = {n["id"] for n in sub["nodes"]}
        assert {"C1", "C2", "D1"}.issubset(node_ids)
        assert len(sub["edges"]) >= 2


@pytest.mark.skipif(not METRICS_PATH.exists(), reason="run scripts/ingest_graph.py first")
class TestIngestedScenarioGraph:
    @classmethod
    def setup_class(cls):
        with open(METRICS_PATH) as f:
            cls.metrics = json.load(f)

    def test_ran_on_real_graph_backend(self):
        assert self.metrics["backend"] == "kuzu"
        assert self.metrics["ingest"]["customers"] > 0
        assert self.metrics["ingest"]["links"] > 0

    def test_planted_fraud_ring_detected(self):
        """Scenario 3 planted a 6-customer ring sharing account+device+email.
        The graph query must find it."""
        sc3 = self.metrics["per_scenario"]["FS-SC-003"]
        assert sc3["expected_risk_level"] == "HIGH"
        assert sc3["max_strong_connections"] >= 5
        assert sc3["any_customer_connected"] is True

    def test_shared_address_scenario_detected(self):
        sc6 = self.metrics["per_scenario"]["FS-SC-006"]
        assert sc6["max_strong_connections"] >= 3

    def test_legitimate_scenario_not_flagged(self):
        """Scenario 1 is a clean single case -- zero strong connections.
        A graph layer that connects everything to everything is useless."""
        sc1 = self.metrics["per_scenario"]["FS-SC-001"]
        assert sc1["expected_risk_level"] == "LOW"
        assert sc1["max_strong_connections"] == 0

    def test_connectivity_separates_risk_levels(self):
        """HIGH-risk scenarios must show more graph connectivity than LOW."""
        by_risk = self.metrics["connectivity_by_expected_risk"]
        assert by_risk["HIGH"]["mean_strong_connections"] > by_risk["LOW"]["mean_strong_connections"]
        assert by_risk["HIGH"]["mean_strong_connections"] > by_risk["MEDIUM"]["mean_strong_connections"]

    def test_graph_alone_misses_document_only_fraud(self):
        """
        HONEST LIMITATION UNDER TEST: Scenario 5 (duplicate invoice reused
        across 3 claims) is HIGH risk but has ZERO shared entities -- the
        graph layer cannot see it at all. This is the concrete evidence
        that no single signal is sufficient, and the reason risk fusion
        exists. If this ever starts passing with connections > 0, the
        scenario data changed and this claim needs revisiting.
        """
        sc5 = self.metrics["per_scenario"]["FS-SC-005"]
        assert sc5["expected_risk_level"] == "HIGH"
        assert sc5["max_strong_connections"] == 0, \
            "Scenario 5 should be invisible to graph analysis (document-only signal)"

    def test_weak_signals_reported_separately(self):
        assert "weak_signal_clusters_reported_separately" in self.metrics
        assert "note_strong_vs_weak" in self.metrics

    def test_example_subgraph_generated_for_dashboard(self):
        sub = self.metrics["example_subgraph_FS-SC-003"]
        assert len(sub["nodes"]) > 1
        assert len(sub["edges"]) > 1


class TestNeo4jBackendCode:
    """Structural validation only -- no Neo4j server in this environment."""

    def test_neo4j_backend_raises_clear_error_when_driver_missing(self):
        from app.graph.store import Neo4jGraphStore
        import importlib.util
        if importlib.util.find_spec("neo4j") is not None:
            pytest.skip("neo4j driver installed -- error path not applicable")
        with pytest.raises(ImportError) as exc:
            Neo4jGraphStore()
        assert "pip install neo4j" in str(exc.value)

    def test_both_backends_implement_same_interface(self):
        from app.graph.store import KuzuGraphStore, Neo4jGraphStore, GraphStore
        for method in ["initialize_schema", "add_customer", "add_entity", "link", "query", "close"]:
            assert hasattr(KuzuGraphStore, method)
            assert hasattr(Neo4jGraphStore, method)
        assert issubclass(KuzuGraphStore, GraphStore)
        assert issubclass(Neo4jGraphStore, GraphStore)
