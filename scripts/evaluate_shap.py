"""
Phase 10a -- SHAP evaluation script.

Runs SHAP against the REAL trained XGBoost model (Phase 3's best model)
and REAL IEEE-CIS test data: global feature importance over a sample,
plus per-case explanations for one true-positive fraud case and one
true-negative legitimate case, so the explanations can be sanity-checked
against ground truth.
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.explainability.shap_explainer import TabularExplainer  # noqa: E402

OUT_PATH = REPO_ROOT / "docs" / "shap_metrics.json"


def main():
    test_df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "ieee_cis_test.parquet")
    sample = test_df.sample(n=300, random_state=42)

    results = {}
    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        print(f"Explaining {model_name}...")
        explainer = TabularExplainer(model_name)

        global_importance = explainer.explain_batch_summary(sample, top_k=10)

        # one confirmed fraud case, one confirmed legitimate case, so the
        # explanation can be checked against reality, not just produced
        fraud_row = test_df[test_df["isFraud"] == 1].sample(n=1, random_state=1)
        legit_row = test_df[test_df["isFraud"] == 0].sample(n=1, random_state=1)

        fraud_explanation = explainer.explain_row(fraud_row, top_k=5)
        legit_explanation = explainer.explain_row(legit_row, top_k=5)

        results[model_name] = {
            "global_importance_sample_size": 300,
            "global_feature_importance": global_importance["global_feature_importance"],
            "example_confirmed_fraud_case": {
                "transaction_id": int(fraud_row["TransactionID"].iloc[0]),
                **fraud_explanation,
            },
            "example_confirmed_legitimate_case": {
                "transaction_id": int(legit_row["TransactionID"].iloc[0]),
                **legit_explanation,
            },
        }
        print(f"  predicted P(fraud) on the confirmed-fraud example: {fraud_explanation['predicted_fraud_probability']}")
        print(f"  predicted P(fraud) on the confirmed-legit example: {legit_explanation['predicted_fraud_probability']}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
