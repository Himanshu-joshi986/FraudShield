"""
Phase 10a -- SHAP explainability for the tabular baseline models.

VERIFIED WORKING: SHAP 0.52.0, run against the ACTUAL trained Phase-3
model artifacts (models/baseline/*.joblib) and real IEEE-CIS test data.
Uses shap.TreeExplainer for XGBoost/RandomForest (exact, fast, no
approximation) and shap.LinearExplainer for Logistic Regression.

Per Section 27: displays top positive/negative features with contribution
values -- not just a bare risk score.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models" / "baseline"
PROCESSED = REPO_ROOT / "data" / "processed"

# must exactly match scripts/train_baseline.py's feature construction --
# a SHAP explanation is only meaningful if it explains the SAME features
# the model was actually trained on.
IEEE_NUMERIC_COLS = (
    ["TransactionAmt_log", "txn_hour", "txn_dayofweek", "card1_freq", "addr1_freq", "has_identity"]
    + [f"C{i}" for i in range(1, 15)]
    + ["D1", "D2", "D3", "D4", "D10", "D15"]
)
IEEE_CATEGORICAL_COLS = ["ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]


def _build_feature_matrix(df: pd.DataFrame, imputer, encoder: OneHotEncoder, feature_names: list[str]):
    df = df.copy()
    for c in IEEE_CATEGORICAL_COLS:
        df[c] = df[c].astype("object").fillna("missing")

    X_num = imputer.transform(df[IEEE_NUMERIC_COLS])
    X_cat = encoder.transform(df[IEEE_CATEGORICAL_COLS])
    X = np.hstack([X_num, X_cat]).astype("float32")
    return pd.DataFrame(X, columns=feature_names, index=df.index)


class TabularExplainer:
    """Loads a trained Phase-3 model bundle and explains predictions
    against it using the SAME feature pipeline that trained it."""

    def __init__(self, model_name: str = "xgboost"):
        bundle_path = MODELS_DIR / f"ieee_cis_{model_name}.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"{bundle_path} not found -- run scripts/train_baseline.py first")

        self.bundle = joblib.load(bundle_path)
        self.model = self.bundle["model"]
        self.imputer = self.bundle["imputer"]
        self.encoder = self.bundle["encoder"]
        self.feature_names = self.bundle["feature_names"]
        self.model_name = model_name

        if model_name in ("xgboost", "random_forest"):
            self.explainer = shap.TreeExplainer(self.model)
        elif model_name == "logistic_regression":
            self.scaler = self.bundle["scaler"]
            # LinearExplainer needs a background distribution
            train_sample = pd.read_parquet(PROCESSED / "ieee_cis_train.parquet").sample(
                n=500, random_state=42
            )
            X_bg = _build_feature_matrix(train_sample, self.imputer, self.encoder, self.feature_names)
            X_bg_scaled = self.scaler.transform(X_bg)
            self.explainer = shap.LinearExplainer(self.model, X_bg_scaled)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def explain_row(self, row_df: pd.DataFrame, top_k: int = 5) -> dict:
        """row_df: single-row DataFrame with the raw IEEE-CIS columns
        (same shape as ieee_cis_{split}.parquet)."""
        X = _build_feature_matrix(row_df, self.imputer, self.encoder, self.feature_names)

        if self.model_name == "logistic_regression":
            X_input = self.scaler.transform(X)
            proba = float(self.model.predict_proba(X_input)[0, 1])
            shap_values = self.explainer.shap_values(X_input)
        else:
            proba = float(self.model.predict_proba(X)[0, 1])
            raw = self.explainer.shap_values(X)
            # TreeExplainer returns either a 2D array (binary, class-1
            # contributions) or a 3D array [n_samples, n_features,
            # n_classes] depending on model/SHAP version -- handle both.
            shap_values = raw[:, :, 1] if isinstance(raw, np.ndarray) and raw.ndim == 3 else raw

        contributions = list(zip(self.feature_names, shap_values[0], X.iloc[0].values))
        contributions.sort(key=lambda c: abs(c[1]), reverse=True)

        top = contributions[:top_k]
        return {
            "model": self.model_name,
            "predicted_fraud_probability": round(proba, 4),
            "top_contributions": [
                {"feature": name, "shap_value": round(float(val), 4), "feature_value": round(float(fval), 4),
                 "direction": "increases_risk" if val > 0 else "decreases_risk"}
                for name, val, fval in top
            ],
        }

    def explain_batch_summary(self, df: pd.DataFrame, top_k: int = 10) -> dict:
        """Global feature importance (mean |SHAP value|) over a sample --
        answers 'what does this model generally weight most heavily',
        distinct from a single case's explanation."""
        X = _build_feature_matrix(df, self.imputer, self.encoder, self.feature_names)

        if self.model_name == "logistic_regression":
            X_input = self.scaler.transform(X)
            shap_values = self.explainer.shap_values(X_input)
        else:
            raw = self.explainer.shap_values(X)
            shap_values = raw[:, :, 1] if isinstance(raw, np.ndarray) and raw.ndim == 3 else raw

        mean_abs = np.abs(shap_values).mean(axis=0)
        ranked = sorted(zip(self.feature_names, mean_abs), key=lambda c: c[1], reverse=True)
        return {
            "model": self.model_name, "n_samples": len(df),
            "global_feature_importance": [
                {"feature": name, "mean_abs_shap": round(float(val), 4)} for name, val in ranked[:top_k]
            ],
        }
