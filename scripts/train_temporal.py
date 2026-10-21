"""Train and evaluate the per-transaction LSTM sequence classifier."""
import argparse
import json
import random
import sys
import time
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
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

FEATURE_DIR = REPO_ROOT / "data" / "processed" / "temporal"
MODEL_DIR = REPO_ROOT / "models" / "temporal"
METRICS_PATH = REPO_ROOT / "docs" / "temporal_lstm_metrics.json"


def load_sequences(split: str, seq_len: int):
    from app.ml.temporal.sequence_model import build_sequences

    features = pd.read_parquet(FEATURE_DIR / f"{split}_temporal_features.parquet")
    x, y = build_sequences(features, seq_len=seq_len)
    return features, x, y


def standardize_sequences(train_x, *other_x):
    """Fit per-feature scaling on training transactions only, preserving zero padding."""
    valid_train_rows = ~np.all(train_x == 0, axis=-1)
    train_rows = train_x[valid_train_rows]
    scaler_mean = train_rows.mean(axis=0)
    scaler_scale = train_rows.std(axis=0)
    scaler_scale[scaler_scale == 0] = 1.0

    scaled = []
    for values in (train_x, *other_x):
        padding = np.all(values == 0, axis=-1, keepdims=True)
        transformed = (values - scaler_mean) / scaler_scale
        transformed[np.broadcast_to(padding, transformed.shape)] = 0.0
        scaled.append(transformed.astype(np.float32, copy=False))
    return (scaler_mean, scaler_scale, *scaled)


def evaluate(model, x, y, device, batch_size):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).to(device)
            probabilities.append(model.predict_proba(batch).cpu().numpy())
    probability = np.concatenate(probabilities)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from app.ml.temporal.sequence_model import (
        FraudLSTM,
        SEQUENCE_FEATURE_COLS,
        class_weighted_loss,
    )

    _, x_train, y_train = load_sequences("train", args.seq_len)
    _, x_val, y_val = load_sequences("val", args.seq_len)
    _, x_test, y_test = load_sequences("test", args.seq_len)
    if y_train.sum() == 0 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Train must contain fraud and validation/test must each contain both classes.")

    mean, scale, x_train, x_val, x_test = standardize_sequences(x_train, x_val, x_test)
    model = FraudLSTM(
        n_features=x_train.shape[-1],
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    criterion = class_weighted_loss(y_train).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_pr_auc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)

        val_metrics = evaluate(model, x_val, y_val, device, args.batch_size)
        history.append({
            "epoch": epoch,
            "train_loss": round(total_loss / len(y_train), 6),
            "val_pr_auc": val_metrics["pr_auc"],
        })
        print(f"epoch {epoch:02d} | loss {total_loss / len(y_train):.4f} "
              f"| val PR-AUC {val_metrics['pr_auc']:.4f}")
        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a selected validation checkpoint.")
    model.load_state_dict(best_state)
    results = {
        "architecture": "FraudLSTM (not a Temporal Fusion Transformer)",
        "device": str(device),
        "hyperparameters": vars(args),
        "feature_names": SEQUENCE_FEATURE_COLS,
        "samples": {
            "train": {"n_samples": len(y_train), "n_positive": int(y_train.sum())},
            "val": {"n_samples": len(y_val), "n_positive": int(y_val.sum())},
            "test": {"n_samples": len(y_test), "n_positive": int(y_test.sum())},
        },
        "val": evaluate(model, x_val, y_val, device, args.batch_size),
        "test": evaluate(model, x_test, y_test, device, args.batch_size),
        "train_time_sec": round(time.time() - started, 1),
        "training_history": history,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "n_features": int(x_train.shape[-1]),
        "hidden_size": args.hidden,
        "num_layers": args.layers,
        "dropout": args.dropout,
        "seq_len": args.seq_len,
        "feature_names": results["feature_names"],
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
    }, MODEL_DIR / "fraud_lstm.pt")
    METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Test results:", json.dumps(results["test"], indent=2))
    print(f"Saved model -> {MODEL_DIR / 'fraud_lstm.pt'}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
