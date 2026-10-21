"""
Phase 3 -- Baseline fraud models, trained and evaluated on the real,
processed IEEE-CIS and carclaims data from Phase 2.

Single CPU core is available in this environment (checked via
os.cpu_count()), so model sizes are kept modest (bounded n_estimators /
max_depth) to keep training time reasonable -- this is a documented
resource constraint, not a modeling choice made for its own sake.

Reports the full metric set Section 6 requires for imbalanced fraud
data (precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, FPR,
fraud-class recall) -- never just accuracy, which is meaningless at
~3.5%/~6% fraud prevalence (a model that always predicts "not fraud"
would score >95% accuracy while catching zero fraud).
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "baseline"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = REPO_ROOT / "docs"


# ============================================================ IEEE-CIS ====

IEEE_NUMERIC_COLS = (
    ["TransactionAmt_log", "txn_hour", "txn_dayofweek", "card1_freq", "addr1_freq", "has_identity"]
    + [f"C{i}" for i in range(1, 15)]
    + ["D1", "D2", "D3", "D4", "D10", "D15"]  # D-cols with < 55% missingness only
)
IEEE_CATEGORICAL_COLS = ["ProductCD", "card4", "card6", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]


def build_ieee_features(train, val, test):
    for df in (train, val, test):
        for c in IEEE_CATEGORICAL_COLS:
            df[c] = df[c].astype("object").fillna("missing")

    imputer = SimpleImputer(strategy="median")
    X_num_train = imputer.fit_transform(train[IEEE_NUMERIC_COLS])
    X_num_val = imputer.transform(val[IEEE_NUMERIC_COLS])
    X_num_test = imputer.transform(test[IEEE_NUMERIC_COLS])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_train = encoder.fit_transform(train[IEEE_CATEGORICAL_COLS])
    X_cat_val = encoder.transform(val[IEEE_CATEGORICAL_COLS])
    X_cat_test = encoder.transform(test[IEEE_CATEGORICAL_COLS])

    X_train = np.hstack([X_num_train, X_cat_train]).astype("float32")
    X_val = np.hstack([X_num_val, X_cat_val]).astype("float32")
    X_test = np.hstack([X_num_test, X_cat_test]).astype("float32")

    feature_names = IEEE_NUMERIC_COLS + list(encoder.get_feature_names_out(IEEE_CATEGORICAL_COLS))

    return (X_train, X_val, X_test), (imputer, encoder), feature_names


def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall_fraud_class": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "false_positive_rate": round(float(fpr), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y)), "n_positive": int(y.sum()),
    }


def run_ieee_cis():
    print("=" * 70, "\nIEEE-CIS baseline models\n", "=" * 70)
    train = pd.read_parquet(PROCESSED / "ieee_cis_train.parquet")
    val = pd.read_parquet(PROCESSED / "ieee_cis_val.parquet")
    test = pd.read_parquet(PROCESSED / "ieee_cis_test.parquet")

    y_train, y_val, y_test = train["isFraud"].values, val["isFraud"].values, test["isFraud"].values

    (X_train, X_val, X_test), (imputer, encoder), feature_names = build_ieee_features(train, val, test)
    print(f"Feature matrix: {X_train.shape[1]} features, {X_train.shape[0]} train rows")

    results = {}

    # --- Logistic Regression ---
    t0 = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(class_weight="balanced", max_iter=500)
    lr.fit(X_train_s, y_train)
    results["logistic_regression"] = {
        "val": evaluate(lr, X_val_s, y_val), "test": evaluate(lr, X_test_s, y_test),
        "train_time_sec": round(time.time() - t0, 1),
    }
    joblib.dump({"model": lr, "scaler": scaler, "imputer": imputer, "encoder": encoder,
                 "feature_names": feature_names}, MODELS_DIR / "ieee_cis_logistic_regression.joblib")
    print("LogisticRegression:", results["logistic_regression"]["test"])

    # --- Random Forest (bounded size: single CPU core available) ---
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=20,
                                 class_weight="balanced", n_jobs=1, random_state=42)
    rf.fit(X_train, y_train)
    results["random_forest"] = {
        "val": evaluate(rf, X_val, y_val), "test": evaluate(rf, X_test, y_test),
        "train_time_sec": round(time.time() - t0, 1),
    }
    joblib.dump({"model": rf, "imputer": imputer, "encoder": encoder,
                 "feature_names": feature_names}, MODELS_DIR / "ieee_cis_random_forest.joblib")
    print("RandomForest:", results["random_forest"]["test"])

    # --- XGBoost ---
    t0 = time.time()
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        n_jobs=1, random_state=42, tree_method="hist",
    )
    xgb.fit(X_train, y_train)
    results["xgboost"] = {
        "val": evaluate(xgb, X_val, y_val), "test": evaluate(xgb, X_test, y_test),
        "train_time_sec": round(time.time() - t0, 1),
    }
    joblib.dump({"model": xgb, "imputer": imputer, "encoder": encoder,
                 "feature_names": feature_names}, MODELS_DIR / "ieee_cis_xgboost.joblib")
    print("XGBoost:", results["xgboost"]["test"])

    with open(DOCS_DIR / "ieee_cis_baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================ carclaims ===

def run_carclaims():
    print("\n" + "=" * 70, "\ncarclaims baseline models\n", "=" * 70)
    train = pd.read_parquet(PROCESSED / "carclaims_train.parquet")
    val = pd.read_parquet(PROCESSED / "carclaims_val.parquet")
    test = pd.read_parquet(PROCESSED / "carclaims_test.parquet")

    y_train, y_val, y_test = train["fraud_label"].values, val["fraud_label"].values, test["fraud_label"].values

    feature_cols = [c for c in train.columns if c not in ("claim_id", "fraud_label", "data_source", "PolicyNumber")]
    categorical_cols = [c for c in feature_cols if str(train[c].dtype) == "category"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    for df in (train, val, test):
        for c in categorical_cols:
            df[c] = df[c].astype("object").fillna("missing")

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_train = encoder.fit_transform(train[categorical_cols])
    X_cat_val = encoder.transform(val[categorical_cols])
    X_cat_test = encoder.transform(test[categorical_cols])

    X_train = np.hstack([train[numeric_cols].values, X_cat_train]).astype("float32")
    X_val = np.hstack([val[numeric_cols].values, X_cat_val]).astype("float32")
    X_test = np.hstack([test[numeric_cols].values, X_cat_test]).astype("float32")
    feature_names = numeric_cols + list(encoder.get_feature_names_out(categorical_cols))
    print(f"Feature matrix: {X_train.shape[1]} features, {X_train.shape[0]} train rows")

    results = {}

    t0 = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(class_weight="balanced", max_iter=500)
    lr.fit(X_train_s, y_train)
    results["logistic_regression"] = {
        "val": evaluate(lr, X_val_s, y_val), "test": evaluate(lr, X_test_s, y_test),
        "train_time_sec": round(time.time() - t0, 1),
    }
    joblib.dump({"model": lr, "scaler": scaler, "encoder": encoder, "numeric_cols": numeric_cols,
                 "categorical_cols": categorical_cols, "feature_names": feature_names},
                MODELS_DIR / "carclaims_logistic_regression.joblib")
    print("LogisticRegression:", results["logistic_regression"]["test"])

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5,
                                 class_weight="balanced", n_jobs=1, random_state=42)
    rf.fit(X_train, y_train)
    results["random_forest"] = {
        "val": evaluate(rf, X_val, y_val), "test": evaluate(rf, X_test, y_test),
        "train_time_sec": round(time.time() - t0, 1),
    }
    joblib.dump({"model": rf, "encoder": encoder, "numeric_cols": numeric_cols,
                 "categorical_cols": categorical_cols, "feature_names": feature_names},
                MODELS_DIR / "carclaims_random_forest.joblib")
    print("RandomForest:", results["random_forest"]["test"])

    with open(DOCS_DIR / "carclaims_baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    ieee_results = run_ieee_cis()
    carclaims_results = run_carclaims()
    print("\nAll baseline models trained and saved to", MODELS_DIR)
