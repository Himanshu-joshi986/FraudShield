"""
Phase 5b -- LSTM sequence model for temporal fraud detection
(REQUIRES PyTorch).

Training and evaluation are available through scripts/train_temporal.py.
The measured artifact and metrics are saved under models/temporal/ and docs/.

    The feature-engineering + IsolationForest version
    (ml/temporal/velocity_detector.py) IS torch-free, WAS run on the real
    IEEE-CIS data, and its measured results are in docs/EXPERIMENTS.md §6.

WHAT THIS IS, PRECISELY (Section 15 requires this honesty):
This is an **LSTM sequence classifier**.
It is NOT a Temporal Fusion Transformer.
Do not describe it as a TFT in your report. A TFT adds
variable-selection networks, static covariate encoders, and
interpretable multi-head attention over a multi-horizon forecasting
objective -- substantially more machinery, aimed at a different problem
shape (multi-horizon forecasting with known future inputs), and not
clearly justified for binary per-transaction fraud classification on this
dataset. If you later want a genuine TFT, use `pytorch-forecasting`'s
implementation rather than relabeling this one.

Sequence definition: for each transaction, the model sees the last
`seq_len` transactions by the same proxy entity (card1+addr1+email),
ordered in time, and classifies whether the final transaction is fraud.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


SEQUENCE_FEATURE_COLS = [
    "TransactionAmt_log", "seconds_since_prev", "amt_zscore_vs_own_history",
    "txns_in_trailing_hour", "is_unusual_hour", "txn_hour",
]


class FraudLSTM(nn.Module):
    """
    LSTM over an entity's recent transaction sequence.

    Why a sequence model adds something the Phase-3 tabular baseline
    can't: the baseline sees one transaction in isolation. This sees the
    *shape* of an entity's recent behaviour -- a slow ramp, a sudden
    burst, a dormant account waking up -- which is what velocity fraud
    actually looks like.
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]              # final layer's last hidden state
        return self.classifier(self.dropout(last_hidden))

    @torch.no_grad()
    def predict_proba(self, x):
        """Returns P(fraud), matching the contract risk fusion expects."""
        self.eval()
        return torch.softmax(self.forward(x), dim=1)[:, 1]


def build_sequences(df, seq_len: int = 10, feature_cols=None):
    """
    Turns per-transaction rows into per-transaction *sequences*.

    For each transaction i belonging to entity e, the sequence is the
    `seq_len` most recent transactions by e up to and including i,
    left-padded with zeros when the entity has fewer than seq_len
    transactions so far.

    CRITICAL -- no leakage: only transactions at or before i are ever
    included. A sequence never contains an entity's future transactions.

    Input `df` must already have been through
    ml/temporal/velocity_detector.engineer_temporal_features().

    Returns: (X, y) where X is (n, seq_len, n_features), y is (n,)
    """
    feature_cols = feature_cols or SEQUENCE_FEATURE_COLS

    df = df.sort_values(["proxy_uid", "TransactionDT"]).reset_index(drop=True)
    feats = df[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype="float32")
    labels = df["isFraud"].to_numpy(dtype="int64")
    uids = df["proxy_uid"].to_numpy()

    n, n_features = feats.shape
    X = np.zeros((n, seq_len, n_features), dtype="float32")

    start = 0
    for i in range(1, n + 1):
        if i == n or uids[i] != uids[start]:
            # rows [start, i) belong to one entity, already time-sorted
            for pos in range(start, i):
                window_start = max(start, pos - seq_len + 1)
                window = feats[window_start:pos + 1]
                X[pos, seq_len - len(window):] = window   # left-pad
            start = i

    return X, labels


def class_weighted_loss(y: np.ndarray) -> nn.Module:
    """Same imbalance handling as the GNN -- without it the model predicts
    'not fraud' everywhere and scores 96% accuracy with 0% recall."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    weights = torch.tensor([1.0, n_neg / max(1, n_pos)], dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weights)
