"""Generate and persist one real GNNExplainer result from the held-out graph."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-idx", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    from app.explainability.gnn_explainer import explain_and_map_to_transactions
    from app.ml.gnn.model import FraudSAGE, build_pyg_data

    model_dir = REPO_ROOT / "models" / "gnn"
    checkpoint = torch.load(
        model_dir / "gnn_fraudsage.pt", map_location="cpu", weights_only=True
    )
    if checkpoint.get("architecture") != "FraudSAGE":
        raise ValueError("Checkpoint was not trained with this repository's FraudSAGE model.")
    features = pd.read_parquet(REPO_ROOT / "data" / "processed" / "graph" / "test_node_features.parquet")
    labels = pd.read_parquet(REPO_ROOT / "data" / "processed" / "graph" / "test_node_labels.parquet")
    edges = pd.read_parquet(REPO_ROOT / "data" / "processed" / "graph" / "test_edges.parquet")
    metadata = pd.read_parquet(REPO_ROOT / "data" / "processed" / "graph" / "test_node_meta.parquet")
    if list(features.columns) != checkpoint["feature_names"]:
        raise ValueError("Held-out graph features differ from the trained checkpoint feature order.")

    scaled_features = (
        features.to_numpy(dtype="float32")
        - torch.tensor(checkpoint["feature_mean"], dtype=torch.float32).numpy()
    ) / torch.tensor(checkpoint["feature_scale"], dtype=torch.float32).numpy()
    data = build_pyg_data(pd.DataFrame(scaled_features, columns=features.columns), labels, edges)
    model = FraudSAGE(
        in_channels=checkpoint["in_channels"],
        hidden_channels=checkpoint["hidden_channels"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    if args.node_idx is None:
        degree = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
        candidates = torch.where((data.y == 1) & (degree > 0))[0]
        if candidates.numel() == 0:
            raise ValueError("No fraud-labeled test node with a graph neighbor is available to explain.")
        probabilities = model.predict_proba(data.x, data.edge_index)
        args.node_idx = int(candidates[torch.argmax(probabilities[candidates])])
    if not 0 <= args.node_idx < data.num_nodes:
        raise ValueError(f"node index {args.node_idx} is outside the test graph.")

    result = explain_and_map_to_transactions(model, data, metadata, args.node_idx)
    result["prediction_probability"] = round(
        float(model.predict_proba(data.x, data.edge_index)[args.node_idx]), 6
    )
    result["ground_truth_is_fraud"] = int(data.y[args.node_idx])
    result["explainer_epochs"] = args.epochs
    result["feature_names"] = checkpoint["feature_names"]
    output = REPO_ROOT / "docs" / "gnn_explanations.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved explanation -> {output}")


if __name__ == "__main__":
    main()
