"""Train and evaluate the repository's GraphSAGE fraud classifier.

Usage:
    python3 scripts/train_gnn.py --epochs 100 --hidden 64

Prerequisite: run the torch-free graph builder first:
    python3 backend/app/ml/gnn/graph_builder.py --sample-size 50000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

GRAPH_DIR = REPO_ROOT / "data" / "processed" / "graph"
MODELS_DIR = REPO_ROOT / "models" / "gnn"
DOCS_DIR = REPO_ROOT / "docs"
def load_split(split_name):
    features = pd.read_parquet(GRAPH_DIR / f"{split_name}_node_features.parquet")
    labels = pd.read_parquet(GRAPH_DIR / f"{split_name}_node_labels.parquet")
    edges = pd.read_parquet(GRAPH_DIR / f"{split_name}_edges.parquet")
    return features, labels, edges


def standardize_features(train_features, *other_features):
    """Fit on train only so held-out graph statistics do not affect scaling."""
    mean = train_features.mean(axis=0).to_numpy(dtype="float32")
    scale = train_features.std(axis=0).to_numpy(dtype="float32").copy()
    scale[~(scale > 0)] = 1.0
    transformed = []
    for features in (train_features, *other_features):
        values = (features.to_numpy(dtype="float32") - mean) / scale
        transformed.append(pd.DataFrame(values, columns=features.columns))
    return mean, scale, transformed


def evaluate_gnn(model, data, torch):
    """Same metric set as the Phase-3 baselines, so results are directly comparable."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        average_precision_score, confusion_matrix,
    )

    proba = model.predict_proba(data.x, data.edge_index).cpu().numpy()
    y_true = data.y.cpu().numpy()
    preds = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall_fraud_class": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)), "n_positive": int(y_true.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15, help="early-stopping patience on val PR-AUC")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit(
            "PyTorch is not installed.\n"
            "See docs/RUN_GNN.md for install instructions:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install torch_geometric"
        )

    from app.ml.gnn.model import FraudSAGE, build_pyg_data, class_weighted_loss

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = {}
    feature_names = None
    loaded = {}
    for name in ["train", "val", "test"]:
        features, labels, edges = load_split(name)
        loaded[name] = (features, labels, edges)
    feature_names = list(loaded["train"][0].columns)
    feature_mean, feature_scale, scaled_features = standardize_features(
        loaded["train"][0], loaded["val"][0], loaded["test"][0]
    )
    for name, features in zip(["train", "val", "test"], scaled_features):
        _, labels, edges = loaded[name]
        splits[name] = build_pyg_data(features, labels, edges).to(device)
        print(f"{name}: {splits[name].num_nodes} nodes, {splits[name].num_edges} edges")

    model = FraudSAGE(
        in_channels=splits["train"].num_node_features,
        hidden_channels=args.hidden,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = class_weighted_loss(splits["train"].y).to(device)

    best_val_pr_auc, best_state, epochs_without_improvement = -1.0, None, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(splits["train"].x, splits["train"].edge_index)
        loss = criterion(out, splits["train"].y)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate_gnn(model, splits["val"], torch)
            print(f"epoch {epoch:3d} | loss {loss.item():.4f} | "
                  f"val PR-AUC {val_metrics['pr_auc']:.4f} | val ROC-AUC {val_metrics['roc_auc']:.4f}")

            if val_metrics["pr_auc"] > best_val_pr_auc:
                best_val_pr_auc = val_metrics["pr_auc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 5
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no val PR-AUC improvement)")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    results = {
        "val": evaluate_gnn(model, splits["val"], torch),
        "test": evaluate_gnn(model, splits["test"], torch),
        "train_time_sec": round(time.time() - t0, 1),
        "hyperparameters": vars(args),
        "device": str(device),
        "architecture": "FraudSAGE(GraphSAGE)",
        "input_features": int(splits["train"].num_node_features),
        "graph_samples": {
            name: int(data.num_nodes) for name, data in splits.items()
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "in_channels": int(splits["train"].num_node_features),
        "hidden_channels": args.hidden,
        "architecture": "FraudSAGE",
        "dropout": args.dropout,
        "feature_names": feature_names,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "metrics_file": "gnn_metrics.json",
    }
    torch.save(checkpoint, MODELS_DIR / "gnn_fraudsage.pt")

    for output_path in (MODELS_DIR / "gnn_metrics.json", DOCS_DIR / "gnn_metrics.json"):
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\nTest results:", json.dumps(results["test"], indent=2))
    print(f"\nSaved model -> {MODELS_DIR / 'gnn_fraudsage.pt'}")
    print(f"Saved metrics -> {MODELS_DIR / 'gnn_metrics.json'}")


if __name__ == "__main__":
    main()
