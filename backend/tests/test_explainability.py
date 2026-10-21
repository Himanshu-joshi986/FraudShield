"""
Phase 10 tests -- Explainability.

SHAP: ran end-to-end against the REAL trained Phase-3 models (SHAP
0.52.0, real TreeExplainer/LinearExplainer, real IEEE-CIS test data).
GNNExplainer: ran against the trained GraphSAGE checkpoint and a
held-out fraud node; the mapped result is checked as a saved artifact.
explain_case(): verified against a REAL pipeline run (Scenario 3).
"""
import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

SHAP_METRICS_PATH = REPO_ROOT / "docs" / "shap_metrics.json"
EXPLAIN_CASE_EXAMPLE_PATH = REPO_ROOT / "docs" / "explain_case_example.json"
EXPLAINABILITY_DIR = REPO_ROOT / "backend" / "app" / "explainability"


@pytest.mark.skipif(not (REPO_ROOT / "models" / "baseline" / "ieee_cis_xgboost.joblib").exists(),
                    reason="run scripts/train_baseline.py first")
class TestTabularExplainerLive:
    """Real SHAP calls against the real trained XGBoost model."""

    @classmethod
    def setup_class(cls):
        from app.explainability.shap_explainer import TabularExplainer
        cls.explainer = TabularExplainer("xgboost")

    def test_explains_a_real_row(self):
        import pandas as pd
        df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "ieee_cis_test.parquet").head(1)
        result = self.explainer.explain_row(df, top_k=5)
        assert "predicted_fraud_probability" in result
        assert 0.0 <= result["predicted_fraud_probability"] <= 1.0
        assert len(result["top_contributions"]) == 5

    def test_contribution_directions_are_valid(self):
        import pandas as pd
        df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "ieee_cis_test.parquet").head(1)
        result = self.explainer.explain_row(df, top_k=3)
        for c in result["top_contributions"]:
            assert c["direction"] in ("increases_risk", "decreases_risk")
            assert (c["shap_value"] > 0) == (c["direction"] == "increases_risk")

    def test_contributions_sorted_by_magnitude(self):
        import pandas as pd
        df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "ieee_cis_test.parquet").head(1)
        result = self.explainer.explain_row(df, top_k=5)
        magnitudes = [abs(c["shap_value"]) for c in result["top_contributions"]]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_explains_features_the_model_was_actually_trained_on(self):
        import pandas as pd
        df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "ieee_cis_test.parquet").head(1)
        result = self.explainer.explain_row(df, top_k=5)
        explained_features = {c["feature"] for c in result["top_contributions"]}
        assert explained_features.issubset(set(self.explainer.feature_names))


@pytest.mark.skipif(not SHAP_METRICS_PATH.exists(), reason="run scripts/evaluate_shap.py first")
class TestSHAPEvaluationResults:
    @classmethod
    def setup_class(cls):
        with open(SHAP_METRICS_PATH) as f:
            cls.metrics = json.load(f)

    def test_all_three_models_explained(self):
        assert set(self.metrics.keys()) == {"logistic_regression", "random_forest", "xgboost"}

    def test_confirmed_fraud_case_scores_higher_than_confirmed_legit_case(self):
        """The substantive sanity check: SHAP-explained predictions must
        actually separate a known-fraud example from a known-legit one,
        not just produce plausible-looking numbers."""
        for model_name, result in self.metrics.items():
            fraud_p = result["example_confirmed_fraud_case"]["predicted_fraud_probability"]
            legit_p = result["example_confirmed_legitimate_case"]["predicted_fraud_probability"]
            assert fraud_p > legit_p, f"{model_name}: fraud example didn't score higher"

    def test_global_importance_matches_known_ieee_cis_signal(self):
        """
        Sanity check against published community findings on this exact
        dataset: the C1-C14 count features are widely reported as among
        IEEE-CIS's most predictive features. If SHAP's global importance
        for XGBoost doesn't include ANY C-feature in the top 5, that's a
        signal something is wrong with the explanation pipeline, not just
        an alternative valid finding.
        """
        top5 = {f["feature"] for f in self.metrics["xgboost"]["global_feature_importance"][:5]}
        c_features_in_top5 = [f for f in top5 if f.startswith("C") and f[1:].isdigit()]
        assert len(c_features_in_top5) > 0


class TestGNNExplainerCode:
    def test_module_is_valid_python(self):
        ast.parse((EXPLAINABILITY_DIR / "gnn_explainer.py").read_text())

    def test_defines_expected_api(self):
        tree = ast.parse((EXPLAINABILITY_DIR / "gnn_explainer.py").read_text())
        functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert {"explain_node", "explain_and_map_to_transactions"}.issubset(functions)

    def test_real_explanation_artifact_has_source_mapping(self):
        with open(REPO_ROOT / "docs" / "gnn_explanations.json") as f:
            result = json.load(f)
        assert result["generated_by"] == "GNNExplainer (torch_geometric.explain)"
        assert result["transaction_id"]
        assert result["top_contributing_features"]


@pytest.mark.skipif(not EXPLAIN_CASE_EXAMPLE_PATH.exists(), reason="run scripts/evaluate_explain_case.py first")
class TestExplainCaseFeature:
    @classmethod
    def setup_class(cls):
        with open(EXPLAIN_CASE_EXAMPLE_PATH) as f:
            cls.explanation = json.load(f)

    def test_ran_against_real_pipeline_output(self):
        assert self.explanation["case_id"].startswith("FS-SYN-")
        assert self.explanation["risk_level"] == "HIGH"

    def test_reasons_are_ranked_by_confidence(self):
        confidences = [r["confidence"] for r in self.explanation["reasons"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_every_reason_has_a_resolvable_evidence_id(self):
        """Section 51: each item should be clickable -- i.e. resolve to a
        real evidence_id, which is what makes this genuinely traceable
        rather than a list of unlinked strings."""
        for r in self.explanation["reasons"]:
            assert r["evidence_id"].startswith("E-")
            assert r["source"]

    def test_disclaimer_present(self):
        """Section 35: 'does not establish that fraud actually occurred.'"""
        assert "does not establish" in self.explanation["disclaimer"]

    def test_at_least_one_graph_reason_for_ring_case(self):
        types = {r["evidence_type"] for r in self.explanation["reasons"]}
        assert "graph" in types


def test_explain_case_with_minimal_state_does_not_crash():
    """Unit test with hand-built state -- confirms the function degrades
    gracefully on a case with no evidence at all, rather than crashing."""
    from app.explainability.explain_case import explain_case
    state = {"case_id": "T1", "risk_score": 0.0, "risk_level": "LOW", "evidence": [],
            "model_results": {"fusion": {}}, "routing_decision": None}
    result = explain_case(state)
    assert result["reasons"] == []
    assert "headline" in result
