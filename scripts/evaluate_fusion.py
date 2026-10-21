"""Compare XGBoost alone with graph + temporal signals on identical test rows."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
PROCESSED = REPO_ROOT / "data" / "processed"


def metrics(y, probability):
    predicted = (probability >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 4),
        "recall_fraud_class": round(float(recall_score(y, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "pr_auc": round(float(average_precision_score(y, probability)), 4),
        "false_positive_rate": round(float(fp / max(1, fp + tn)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
    }


def main():
    from app.ml.gnn.model import FraudSAGE, build_pyg_data
    from app.ml.temporal.sequence_model import FraudLSTM, build_sequences
    from train_baseline import IEEE_CATEGORICAL_COLS, IEEE_NUMERIC_COLS
    from app.ml.fusion.risk_fusion import DEFAULT_WEIGHTS

    gnn_dir = REPO_ROOT / "models" / "gnn"
    temporal_dir = REPO_ROOT / "models" / "temporal"
    gnn_checkpoint = torch.load(gnn_dir / "gnn_fraudsage.pt", map_location="cpu", weights_only=True)
    if gnn_checkpoint.get("architecture") != "FraudSAGE":
        raise ValueError("GNN checkpoint architecture metadata does not match FraudSAGE.")
    if gnn_checkpoint.get("in_channels") != len(gnn_checkpoint["feature_names"]):
        raise ValueError("GNN checkpoint feature metadata is inconsistent.")

    gnn_features = pd.read_parquet(PROCESSED / "graph" / "test_node_features.parquet")
    gnn_labels = pd.read_parquet(PROCESSED / "graph" / "test_node_labels.parquet")
    gnn_edges = pd.read_parquet(PROCESSED / "graph" / "test_edges.parquet")
    node_meta = pd.read_parquet(PROCESSED / "graph" / "test_node_meta.parquet")
    if list(gnn_features.columns) != gnn_checkpoint["feature_names"]:
        raise ValueError("Current graph feature columns do not match the trained GNN checkpoint.")
    graph_features = (
        gnn_features.to_numpy(dtype=np.float32)
        - np.asarray(gnn_checkpoint["feature_mean"], dtype=np.float32)
    ) / np.asarray(gnn_checkpoint["feature_scale"], dtype=np.float32)
    graph_features = pd.DataFrame(graph_features, columns=gnn_features.columns)
    graph_data = build_pyg_data(graph_features, gnn_labels, gnn_edges)
    gnn = FraudSAGE(
        in_channels=gnn_checkpoint["in_channels"],
        hidden_channels=gnn_checkpoint["hidden_channels"],
        dropout=gnn_checkpoint["dropout"],
    )
    gnn.load_state_dict(gnn_checkpoint["state_dict"], strict=True)
    graph_probability = gnn.predict_proba(graph_data.x, graph_data.edge_index).numpy()

    temporal_checkpoint = torch.load(
        temporal_dir / "fraud_lstm.pt", map_location="cpu", weights_only=True
    )
    temporal_df = pd.read_parquet(
        PROCESSED / "temporal" / "test_temporal_features.parquet"
    )
    sequence_x, _ = build_sequences(temporal_df, seq_len=temporal_checkpoint["seq_len"])
    mean = np.asarray(temporal_checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(temporal_checkpoint["scaler_scale"], dtype=np.float32)
    padding = np.all(sequence_x == 0, axis=-1, keepdims=True)
    sequence_x = (sequence_x - mean) / scale
    sequence_x[np.broadcast_to(padding, sequence_x.shape)] = 0.0
    temporal_model = FraudLSTM(
        n_features=temporal_checkpoint["n_features"],
        hidden_size=temporal_checkpoint["hidden_size"],
        num_layers=temporal_checkpoint["num_layers"],
        dropout=temporal_checkpoint["dropout"],
    )
    temporal_model.load_state_dict(temporal_checkpoint["state_dict"], strict=True)
    temporal_model.eval()
    with torch.no_grad():
        temporal_probability = temporal_model.predict_proba(torch.from_numpy(sequence_x)).numpy()

    rows = pd.read_parquet(PROCESSED / "ieee_cis_test.parquet")
    row_by_id = rows.set_index("TransactionID", drop=False)
    common_ids = node_meta["TransactionID"].tolist()
    if len(set(common_ids)) != len(common_ids):
        raise ValueError("Graph test node metadata contains duplicate TransactionIDs.")
    if not set(common_ids).issubset(set(temporal_df["TransactionID"])):
        raise ValueError("Temporal test features do not cover every GNN test transaction.")
    temporal_position = pd.Series(
        np.arange(len(temporal_df)), index=temporal_df["TransactionID"]
    )
    temporal_indices = temporal_position.loc[common_ids].to_numpy()
    tabular = row_by_id.loc[common_ids].copy()
    if tabular["isFraud"].to_numpy().tolist() != gnn_labels["isFraud"].to_numpy().tolist():
        raise ValueError("GNN and raw test labels are not aligned by TransactionID.")

    bundle = __import__("joblib").load(
        REPO_ROOT / "models" / "baseline" / "ieee_cis_xgboost.joblib"
    )
    for column in IEEE_CATEGORICAL_COLS:
        tabular[column] = tabular[column].astype("object").fillna("missing")
    x_numeric = bundle["imputer"].transform(tabular[IEEE_NUMERIC_COLS])
    x_categorical = bundle["encoder"].transform(tabular[IEEE_CATEGORICAL_COLS])
    tabular_x = np.hstack([x_numeric, x_categorical]).astype(np.float32)
    xgb_probability = bundle["model"].predict_proba(tabular_x)[:, 1]

    y = gnn_labels["isFraud"].to_numpy(dtype=np.int8)
    aligned_graph = graph_probability
    aligned_temporal = temporal_probability[temporal_indices]
    signals = {
        "xgboost": xgb_probability,
        "graphsage": aligned_graph,
        "lstm": aligned_temporal,
    }
    available_weights = {
        "xgboost": DEFAULT_WEIGHTS["tabular_ml_risk"],
        "graphsage": DEFAULT_WEIGHTS["graph_risk"],
        "lstm": DEFAULT_WEIGHTS["temporal_risk"],
    }
    total_weight = sum(available_weights.values())
    fused_probability = sum(
        signals[name] * weight for name, weight in available_weights.items()
    ) / total_weight

    results = {
        "evaluation": {
            "dataset": "IEEE-CIS chronological test split",
            "rows": len(y),
            "positive": int(y.sum()),
            "transaction_id_min": int(min(common_ids)),
            "transaction_id_max": int(max(common_ids)),
            "fusion_weights": available_weights,
            "fusion_normalization": "available signal weights renormalized; document signal omitted",
            "threshold": 0.5,
        },
        "models": {
            "xgboost_only": metrics(y, signals["xgboost"]),
            "graphsage_only": metrics(y, signals["graphsage"]),
            "lstm_only": metrics(y, signals["lstm"]),
            "xgboost_graphsage_lstm_fusion": metrics(y, fused_probability),
        },
    }
    output = REPO_ROOT / "docs" / "fusion_metrics.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved metrics -> {output}")


if __name__ == "__main__":
    main()
