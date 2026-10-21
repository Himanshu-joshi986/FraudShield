"""
Phase 4 tests.

Split deliberately in two:
  - TestGraphConstruction: tests the torch-free graph builder against the
    REAL artifacts it produced from IEEE-CIS. Fully verified.
  - TestGNNModelCode: verifies the PyTorch model module is syntactically
    valid and structurally correct WITHOUT requiring torch to be installed
    (it parses the AST). If torch IS installed, it additionally runs a
    real forward pass on a tiny synthetic graph.

The optional PyTorch model checks run when the runtime dependencies are
available; the checked-in checkpoint is separately validated for shape
and architecture compatibility.
"""
import ast
import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_DIR = REPO_ROOT / "data" / "processed" / "graph"
GNN_DIR = REPO_ROOT / "backend" / "app" / "ml" / "gnn"


@pytest.mark.skipif(not (GRAPH_DIR / "train_graph_stats.json").exists(),
                    reason="run backend/app/ml/gnn/graph_builder.py first")
class TestGraphConstruction:
    @classmethod
    def setup_class(cls):
        with open(GRAPH_DIR / "train_graph_stats.json") as f:
            cls.stats = json.load(f)
        cls.features = pd.read_parquet(GRAPH_DIR / "train_node_features.parquet")
        cls.labels = pd.read_parquet(GRAPH_DIR / "train_node_labels.parquet")
        cls.edges = pd.read_parquet(GRAPH_DIR / "train_edges.parquet")
        cls.meta = pd.read_parquet(GRAPH_DIR / "train_node_meta.parquet")
        with open(GRAPH_DIR / "train_fraud_rings.json") as f:
            cls.rings = json.load(f)

    def test_node_counts_align_across_artifacts(self):
        n = self.stats["n_nodes"]
        assert len(self.features) == n
        assert len(self.labels) == n
        assert len(self.meta) == n

    def test_edges_reference_valid_node_indices(self):
        """An edge pointing at a node index that doesn't exist would crash
        PyG at training time with an opaque error -- catch it here instead."""
        n = self.stats["n_nodes"]
        assert self.edges["src_idx"].max() < n
        assert self.edges["dst_idx"].max() < n
        assert self.edges["src_idx"].min() >= 0
        assert self.edges["dst_idx"].min() >= 0

    def test_no_self_loops(self):
        assert (self.edges["src_idx"] != self.edges["dst_idx"]).all()

    def test_edges_respect_min_shared_entities(self):
        min_shared = self.stats["min_shared_entities"]
        assert (self.edges["shared_entities"] >= min_shared).all()

    def test_features_have_no_nulls(self):
        """PyG will happily propagate NaNs through the whole graph and
        produce NaN loss -- features must be fully imputed before training."""
        assert not self.features.isnull().any().any()

    def test_graph_has_meaningful_connectivity(self):
        """A graph with almost no edges makes the GNN pointless (it would
        degenerate to an MLP on node features)."""
        assert self.stats["n_edges"] > 0
        assert self.stats["avg_degree"] > 0.5

    def test_fraud_rings_extracted(self):
        assert len(self.rings) > 0
        for ring in self.rings[:20]:
            assert ring["size"] >= 3
            assert 0.0 <= ring["known_fraud_rate"] <= 1.0

    def test_ring_membership_carries_fraud_signal(self):
        """
        The substantive question for Phase 4: does graph structure actually
        help? Measured on the real data, transactions inside a ring have a
        HIGHER fraud rate than those outside. If this ever inverts, the
        graph construction is broken (or the signal genuinely isn't there,
        which would itself be worth knowing).
        """
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(len(self.meta)))
        G.add_edges_from(self.edges[["src_idx", "dst_idx"]].itertuples(index=False, name=None))

        in_ring = set()
        for component in nx.connected_components(G):
            if len(component) >= 3:
                in_ring.update(component)

        meta = self.meta.copy()
        meta["in_ring"] = meta.index.isin(in_ring)
        rates = meta.groupby("in_ring")["isFraud"].mean()

        assert True in rates.index and False in rates.index
        assert rates[True] > rates[False], (
            f"ring members ({rates[True]:.4f}) should have a higher fraud rate "
            f"than non-members ({rates[False]:.4f})"
        )

    def test_some_rings_are_entirely_fraudulent(self):
        """The strongest evidence that coordinated rings genuinely exist in
        this data: components where every member transaction is labeled fraud."""
        all_fraud_rings = [r for r in self.rings if r["known_fraud_rate"] == 1.0]
        assert len(all_fraud_rings) > 0


class TestGNNModelCode:
    """Validates the model API and compatibility of the trained artifact."""

    def test_model_module_is_valid_python(self):
        source = (GNN_DIR / "model.py").read_text()
        ast.parse(source)  # raises SyntaxError if malformed

    def test_model_defines_expected_api(self):
        tree = ast.parse((GNN_DIR / "model.py").read_text())
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

        assert "FraudSAGE" in classes
        assert {"forward", "predict_proba"}.issubset(functions)
        assert "build_pyg_data" in functions
        assert "class_weighted_loss" in functions

    def test_train_script_is_valid_python(self):
        source = (REPO_ROOT / "scripts" / "train_gnn.py").read_text()
        ast.parse(source)

    def test_checkpoint_matches_repository_model(self):
        import torch
        checkpoint = torch.load(
            REPO_ROOT / "models" / "gnn" / "gnn_fraudsage.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["architecture"] == "FraudSAGE"
        assert checkpoint["in_channels"] == len(checkpoint["feature_names"])
        assert checkpoint["in_channels"] == len(
            pd.read_parquet(GRAPH_DIR / "train_node_features.parquet").columns
        )

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("torch_geometric") is None,
        reason="torch_geometric is required for an actual GraphSAGE forward pass",
    )
    def test_forward_pass_on_tiny_graph(self):
        """Only runs where torch+PyG ARE installed (i.e. on the user's
        machine). Confirms the model produces correctly-shaped output."""
        import torch
        import sys
        sys.path.insert(0, str(REPO_ROOT / "backend"))
        from app.ml.gnn.model import FraudSAGE

        model = FraudSAGE(in_channels=4, hidden_channels=8)
        x = torch.randn(5, 4)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        out = model(x, edge_index)
        assert out.shape == (5, 2)

        proba = model.predict_proba(x, edge_index)
        assert proba.shape == (5,)
        assert ((proba >= 0) & (proba <= 1)).all()
