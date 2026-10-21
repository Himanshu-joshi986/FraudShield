"""
Phase 5 tests. Same split as Phase 4:
  - TestTemporalFeatures: tests the REAL artifacts produced by the
    torch-free velocity_detector on IEEE-CIS. Fully verified.
  - TestSequenceModelCode: validates the LSTM module without requiring
    torch; runs a real forward pass only if torch IS installed.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPORAL_DIR = REPO_ROOT / "data" / "processed" / "temporal"
MODULE_DIR = REPO_ROOT / "backend" / "app" / "ml" / "temporal"


@pytest.mark.skipif(not (TEMPORAL_DIR / "train_temporal_features.parquet").exists(),
                    reason="run backend/app/ml/temporal/velocity_detector.py first")
class TestTemporalFeatures:
    @classmethod
    def setup_class(cls):
        cls.train = pd.read_parquet(TEMPORAL_DIR / "train_temporal_features.parquet")
        cls.test = pd.read_parquet(TEMPORAL_DIR / "test_temporal_features.parquet")
        with open(REPO_ROOT / "docs" / "temporal_metrics.json") as f:
            cls.metrics = json.load(f)

    def test_temporal_risk_in_valid_range(self):
        for df in (self.train, self.test):
            assert df["temporal_risk"].between(0.0, 1.0).all()

    def test_no_nulls_in_engineered_features(self):
        cols = ["entity_txn_seq", "amt_zscore_vs_own_history", "txns_in_trailing_hour", "is_unusual_hour"]
        for df in (self.train, self.test):
            assert not df[cols].isnull().any().any()

    def test_entity_sequence_numbers_are_sensible(self):
        """entity_txn_seq must start at 1 for each entity's first
        transaction and increase -- if this breaks, every downstream
        rolling feature is wrong."""
        assert self.train["entity_txn_seq"].min() == 1
        assert (self.train["entity_txn_seq"] > 0).all()

    def test_no_backward_looking_leakage(self):
        """
        An entity's very first transaction has no prior history, so its
        amount z-score vs. its own history must be exactly 0 (not a value
        derived from its own amount, which would be target leakage).
        """
        first_txns = self.train[self.train["entity_txn_seq"] == 1]
        assert (first_txns["amt_zscore_vs_own_history"] == 0.0).all()

    def test_velocity_counts_are_positive(self):
        """Every transaction is inside its own trailing window, so the
        count is always >= 1."""
        assert (self.train["txns_in_trailing_hour"] >= 1).all()

    def test_flagged_transactions_have_elevated_fraud_rate(self):
        """
        The substantive Phase-5 question: do the temporal flags carry real
        signal? Measured on the actual data -- flagged transactions should
        have a higher fraud rate than unflagged ones on every split.
        """
        for split_name, split_stats in self.metrics["splits"].items():
            assert split_stats["fraud_rate_among_flagged"] > split_stats["fraud_rate_among_unflagged"], \
                f"{split_name}: temporal flags carry no signal"

    def test_metrics_record_honest_weak_performance(self):
        """
        This is an unsupervised anomaly detector, not a fraud classifier.
        Its ROC-AUC is expected to be modest (~0.55-0.65). This test
        asserts it beats random (>0.5) but deliberately does NOT assert a
        high value -- overstating this layer's standalone accuracy is
        exactly what Section 61 forbids.
        """
        for split_name, split_stats in self.metrics["splits"].items():
            assert split_stats["roc_auc"] > 0.5, f"{split_name} worse than random"

    def test_thresholds_are_recorded_in_metrics(self):
        """Build Rule 19/20: thresholds must be configurable AND documented
        alongside the results they produced."""
        assert "thresholds" in self.metrics
        assert "high_velocity_count" in self.metrics["thresholds"]


class TestSequenceModelCode:
    def test_module_is_valid_python(self):
        ast.parse((MODULE_DIR / "sequence_model.py").read_text())

    def test_defines_expected_api(self):
        tree = ast.parse((MODULE_DIR / "sequence_model.py").read_text())
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "FraudLSTM" in classes
        assert "build_sequences" in functions
        assert {"forward", "predict_proba"}.issubset(functions)

    def test_module_states_it_is_not_a_TFT(self):
        """Section 15: 'Do NOT claim TFT is implemented if the actual
        system only uses an LSTM.' Enforce that disclaimer stays in place."""
        source = (MODULE_DIR / "sequence_model.py").read_text()
        assert "NOT a Temporal Fusion Transformer" in source

    def test_training_script_is_present(self):
        assert (REPO_ROOT / "scripts" / "train_temporal.py").exists()

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("torch") is None,
        reason="PyTorch is required for an actual LSTM forward pass",
    )
    def test_forward_pass_and_sequence_builder(self):
        import sys
        import torch
        sys.path.insert(0, str(REPO_ROOT / "backend"))
        from app.ml.temporal.sequence_model import FraudLSTM, build_sequences

        model = FraudLSTM(n_features=6, hidden_size=8, num_layers=1)
        x = torch.randn(4, 10, 6)
        assert model(x).shape == (4, 2)
        proba = model.predict_proba(x)
        assert proba.shape == (4,) and ((proba >= 0) & (proba <= 1)).all()

        df = pd.DataFrame({
            "proxy_uid": ["a", "a", "a", "b", "b"],
            "TransactionDT": [1, 2, 3, 1, 2],
            "isFraud": [0, 0, 1, 0, 1],
            "TransactionAmt_log": [1.0, 2.0, 3.0, 1.5, 2.5],
            "seconds_since_prev": [np.inf, 1.0, 1.0, np.inf, 1.0],
            "amt_zscore_vs_own_history": [0.0, 0.5, 1.5, 0.0, 0.8],
            "txns_in_trailing_hour": [1, 2, 3, 1, 2],
            "is_unusual_hour": [0, 0, 1, 0, 0],
            "txn_hour": [10, 10, 3, 14, 14],
        })
        X, y = build_sequences(df, seq_len=3)
        assert X.shape == (5, 3, 6)
        assert len(y) == 5
        assert not np.isinf(X).any(), "inf must be sanitized before the LSTM"
