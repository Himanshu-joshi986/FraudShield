"""
Phase 4b -- GraphSAGE fraud-detection GNN (REQUIRES PyTorch + PyTorch Geometric).

This module implements GraphSAGE using PyTorch Geometric. Training,
checkpoint, and held-out evaluation are documented in docs/RUN_GNN.md
and docs/EXPERIMENTS.md.

Architecture: 2-layer GraphSAGE over the transaction-transaction
adjacency graph produced by graph_builder.py. GraphSAGE is chosen over
GAT here because it samples neighbourhoods rather than attending over all
of them, which scales better on a graph with hub nodes -- and because the
spec explicitly says not to force a more complicated architecture than
the data supports.

Node = one transaction (26 features, binary isFraud label).
Edge  = two transactions sharing >= 2 entities (card/addr/email/device).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudSAGE(nn.Module):
    """
    2-layer GraphSAGE node classifier.

    The key idea vs. the Phase-3 baseline models: a transaction's
    representation is built not only from its own features but from the
    features of transactions it shares entities with. A transaction that
    looks unremarkable alone becomes suspicious when its neighbours in the
    graph are fraudulent -- which is exactly the fraud-ring signal the
    tabular baseline cannot see.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.classifier = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.classifier(h)

    @torch.no_grad()
    def predict_proba(self, x, edge_index):
        """Returns P(fraud) per node, matching the sklearn-style contract
        the risk-fusion layer (Phase 17) expects from every model."""
        self.eval()
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=1)[:, 1]


def build_pyg_data(features_df, labels_df, edges_df):
    """
    Converts the torch-free graph_builder.py parquet output into a
    PyTorch Geometric Data object.

    Edges are made bidirectional (GraphSAGE expects an undirected graph
    represented as both directions) and self-loops are added by the
    SAGEConv layer internally, so we don't add them here.
    """
    from torch_geometric.data import Data

    x = torch.tensor(features_df.values, dtype=torch.float32)
    y = torch.tensor(labels_df["isFraud"].values, dtype=torch.long)

    if len(edges_df) > 0:
        src = torch.tensor(edges_df["src_idx"].values, dtype=torch.long)
        dst = torch.tensor(edges_df["dst_idx"].values, dtype=torch.long)
        # bidirectional
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, y=y, edge_index=edge_index)


def class_weighted_loss(y_train: torch.Tensor) -> nn.Module:
    """
    IEEE-CIS is ~3.5% fraud. Without class weighting the model converges
    to predicting 'not fraud' for everything (high accuracy, zero recall
    -- the exact failure mode Section 6 warns about). Weight the fraud
    class inversely to its frequency.
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    pos_weight = n_neg / max(1, n_pos)
    weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)
    return nn.CrossEntropyLoss(weight=weights)
