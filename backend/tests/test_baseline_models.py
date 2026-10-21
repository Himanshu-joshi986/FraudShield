"""
Phase 3 tests. These check the *actual* saved metrics/artifacts from
scripts/train_baseline.py, not mocked values. Bounds are set generously
around what was actually measured (see docs/EXPERIMENTS.md) so these
catch a real regression (e.g. someone breaking feature engineering)
without being so tight they fail on ordinary run-to-run noise.
"""
import json
from pathlib import Path

import joblib
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS = REPO_ROOT / "docs"
MODELS = REPO_ROOT / "models" / "baseline"


@pytest.mark.skipif(not (DOCS / "ieee_cis_baseline_metrics.json").exists(), reason="run scripts/train_baseline.py first")
class TestIEEECISBaselineModels:
    @classmethod
    def setup_class(cls):
        with open(DOCS / "ieee_cis_baseline_metrics.json") as f:
            cls.results = json.load(f)

    def test_all_three_models_trained(self):
        assert set(self.results.keys()) == {"logistic_regression", "random_forest", "xgboost"}

    def test_metrics_not_just_accuracy(self):
        """Section 6 explicitly forbids relying only on accuracy for
        imbalanced fraud data -- verify the full metric set is present."""
        required = {"precision", "recall_fraud_class", "f1", "roc_auc", "pr_auc",
                    "false_positive_rate", "confusion_matrix"}
        for model_name, model_results in self.results.items():
            assert required.issubset(model_results["test"].keys()), model_name

    def test_models_beat_random_chance(self):
        """A model with ROC-AUC <= 0.5 is worthless (no better than a coin
        flip) -- this is the real floor a broken pipeline should be caught by."""
        for model_name, model_results in self.results.items():
            assert model_results["test"]["roc_auc"] > 0.70, \
                f"{model_name} ROC-AUC {model_results['test']['roc_auc']} suspiciously low"

    def test_recall_is_meaningfully_above_zero(self):
        """A model that predicts 'not fraud' for everything gets 0 recall
        but high accuracy -- this is the exact failure mode Section 6 warns
        about. Verify recall is actually catching fraud cases."""
        for model_name, model_results in self.results.items():
            assert model_results["test"]["recall_fraud_class"] > 0.3, model_name

    def test_confusion_matrix_totals_match_sample_count(self):
        for model_name, model_results in self.results.items():
            cm = model_results["test"]["confusion_matrix"]
            total = cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"]
            assert total == model_results["test"]["n_samples"], model_name

    def test_xgboost_outperforms_logistic_regression_on_pr_auc(self):
        """Sanity check on model ordering: a gradient-boosted tree model
        should outperform plain logistic regression on this kind of
        tabular fraud data -- if this flips, something upstream broke."""
        assert self.results["xgboost"]["test"]["pr_auc"] > self.results["logistic_regression"]["test"]["pr_auc"]

    def test_model_artifacts_exist_and_load(self):
        for name in ["logistic_regression", "random_forest", "xgboost"]:
            path = MODELS / f"ieee_cis_{name}.joblib"
            assert path.exists()
            bundle = joblib.load(path)
            assert "model" in bundle and "feature_names" in bundle


@pytest.mark.skipif(not (DOCS / "carclaims_baseline_metrics.json").exists(), reason="run scripts/train_baseline.py first")
class TestCarclaimsBaselineModels:
    @classmethod
    def setup_class(cls):
        with open(DOCS / "carclaims_baseline_metrics.json") as f:
            cls.results = json.load(f)

    def test_both_models_trained(self):
        assert set(self.results.keys()) == {"logistic_regression", "random_forest"}

    def test_models_beat_random_chance(self):
        for model_name, model_results in self.results.items():
            assert model_results["test"]["roc_auc"] > 0.65, model_name

    def test_model_artifacts_exist_and_load(self):
        for name in ["logistic_regression", "random_forest"]:
            path = MODELS / f"carclaims_{name}.joblib"
            assert path.exists()
            bundle = joblib.load(path)
            assert "model" in bundle and "feature_names" in bundle
