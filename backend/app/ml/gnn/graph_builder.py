"""
Phase 4a -- Entity graph construction (NO PyTorch REQUIRED).

This module is deliberately torch-free so it can be built, tested and
run anywhere, including environments without a GPU. It turns IEEE-CIS
transactions into the heterogeneous entity graph that the GNN
(ml/gnn/model.py) trains on, and that the Investigation Agent later
queries for fraud rings.

Graph schema (Section 13 of the spec):

    Transaction --[USES_CARD]--> Card       (card1 + card4/card6 identity)
    Transaction --[FROM_ADDR]--> Address    (addr1)
    Transaction --[USES_EMAIL]--> Email     (P_emaildomain)
    Transaction --[ON_DEVICE]--> Device     (DeviceInfo, when identity data exists)

Transactions are the labeled nodes (isFraud), and the card/address/email/
device nodes are the *shared* entities through which a fraud ring becomes
visible: two transactions that share a card AND a device AND an email are
far more suspicious together than either is alone.

WHY THIS SHAPE: IEEE-CIS has no explicit "customer" column. The standard
approach on this dataset is to treat the (card1, addr1, P_emaildomain)
combination as a proxy identity ("uid"), because transactions sharing all
three are very likely the same real person. We build that proxy uid here
explicitly rather than pretending a customer_id exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED = REPO_ROOT / "data" / "processed"
GRAPH_OUT = REPO_ROOT / "data" / "processed" / "graph"
GRAPH_OUT.mkdir(parents=True, exist_ok=True)

# Node-feature columns carried onto each transaction node for the GNN.
TXN_NODE_FEATURES = (
    ["TransactionAmt_log", "txn_hour", "txn_dayofweek", "card1_freq", "addr1_freq", "has_identity"]
    + [f"C{i}" for i in range(1, 15)]
    + ["D1", "D2", "D3", "D4", "D10", "D15"]
)


def build_proxy_uid(df: pd.DataFrame) -> pd.Series:
    """
    IEEE-CIS has no customer_id. The widely-used proxy identity is the
    combination of card1 + addr1 + P_emaildomain: transactions matching on
    all three are very likely the same person. Returned as a string key.
    """
    return (
        df["card1"].astype("float").fillna(-1).astype("int64").astype(str)
        + "_" + df["addr1"].astype("float").fillna(-1).astype("int64").astype(str)
        + "_" + df["P_emaildomain"].astype("object").fillna("unknown").astype(str)
    )


def build_entity_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a long-format edge list:
        txn_idx | entity_id | relation

    `txn_idx` is the positional index of the transaction within `df`
    (0..n-1), which is what PyTorch Geometric expects for edge_index.
    """
    edges = []
    n = len(df)
    txn_idx = np.arange(n)

    def add(entity_series: pd.Series, prefix: str, relation: str):
        vals = entity_series.astype("object")
        mask = vals.notna() & (vals.astype(str).str.len() > 0)
        edges.append(pd.DataFrame({
            "txn_idx": txn_idx[mask.values],
            "entity_id": prefix + ":" + vals[mask].astype(str).values,
            "relation": relation,
        }))

    add(df["card1"].astype("float").astype("Int64"), "card", "USES_CARD")
    add(df["addr1"].astype("float").astype("Int64"), "addr", "FROM_ADDR")
    add(df["P_emaildomain"], "email", "USES_EMAIL")
    if "DeviceInfo" in df.columns:
        add(df["DeviceInfo"], "device", "ON_DEVICE")

    return pd.concat(edges, ignore_index=True)


def build_transaction_adjacency(edge_df: pd.DataFrame, min_shared_entities: int = 2,
                                 max_entity_degree: int = 500) -> pd.DataFrame:
    """
    Converts transaction<->entity edges into transaction<->transaction
    edges: two transactions are connected if they share at least
    `min_shared_entities` entities (e.g. same card AND same device).

    `max_entity_degree` guards against hub explosion: an entity like the
    email domain "gmail.com" is attached to hundreds of thousands of
    transactions and carries no fraud-ring signal, but would generate a
    quadratic number of edges. Entities above this degree are dropped as
    uninformative hubs. This threshold is configurable and documented
    rather than silently hardcoded.

    Returns: DataFrame[src_idx, dst_idx, shared_entities]
    """
    degree = edge_df.groupby("entity_id")["txn_idx"].transform("size")
    filtered = edge_df[degree <= max_entity_degree]

    from collections import Counter
    from itertools import combinations

    pair_counts: Counter = Counter()
    for entity_id, group in filtered.groupby("entity_id", sort=False):
        idxs = np.sort(group["txn_idx"].to_numpy(copy=True))
        if len(idxs) < 2:
            continue
        pair_counts.update(combinations(idxs.tolist(), 2))

    if not pair_counts:
        return pd.DataFrame(columns=["src_idx", "dst_idx", "shared_entities"])

    pairs = [(a, b, c) for (a, b), c in pair_counts.items() if c >= min_shared_entities]
    if not pairs:
        return pd.DataFrame(columns=["src_idx", "dst_idx", "shared_entities"])
    return pd.DataFrame(pairs, columns=["src_idx", "dst_idx", "shared_entities"])


def build_graph_from_split(split_name: str, sample_size: int | None = None,
                            min_shared_entities: int = 2, max_entity_degree: int = 500):
    """
    Builds and saves the graph artifacts for one split.

    `sample_size` exists because the full 413K-row train split produces a
    very large transaction-adjacency computation. For development /
    limited hardware, pass e.g. sample_size=50000. Document whatever value
    you use when reporting results -- a GNN trained on a 50K sample is not
    comparable to one trained on the full split.
    """
    df = pd.read_parquet(PROCESSED / f"ieee_cis_{split_name}.parquet")
    if sample_size is not None and sample_size < len(df):
        df = df.sort_values("TransactionDT").head(sample_size).copy()
    df = df.reset_index(drop=True)

    df["proxy_uid"] = build_proxy_uid(df)
    edge_df = build_entity_edges(df)
    adjacency = build_transaction_adjacency(
        edge_df, min_shared_entities=min_shared_entities, max_entity_degree=max_entity_degree
    )

    features = df[TXN_NODE_FEATURES].astype("float32")
    features = features.fillna(features.median())
    labels = df["isFraud"].astype("int8")

    features.to_parquet(GRAPH_OUT / f"{split_name}_node_features.parquet", index=False)
    labels.to_frame("isFraud").to_parquet(GRAPH_OUT / f"{split_name}_node_labels.parquet", index=False)
    adjacency.to_parquet(GRAPH_OUT / f"{split_name}_edges.parquet", index=False)
    df[["TransactionID", "proxy_uid", "TransactionDT", "isFraud"]].to_parquet(
        GRAPH_OUT / f"{split_name}_node_meta.parquet", index=False
    )

    stats = {
        "split": split_name,
        "n_nodes": int(len(df)),
        "n_edges": int(len(adjacency)),
        "n_fraud_nodes": int(labels.sum()),
        "fraud_rate_pct": round(float(labels.mean() * 100), 3),
        "n_features": int(features.shape[1]),
        "n_distinct_entities": int(edge_df["entity_id"].nunique()),
        "min_shared_entities": min_shared_entities,
        "max_entity_degree": max_entity_degree,
        "sample_size": sample_size,
        "avg_degree": round(float(2 * len(adjacency) / max(1, len(df))), 3),
    }
    with open(GRAPH_OUT / f"{split_name}_graph_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def extract_fraud_rings(split_name: str, min_ring_size: int = 3):
    """
    Connected-component analysis over the transaction adjacency graph:
    each component of >= min_ring_size transactions is a candidate
    "fraud ring" (a cluster of transactions bound together by shared
    cards/devices/emails/addresses).

    NOTE ON LANGUAGE (Section 43): a connected component is a *candidate
    cluster requiring review*, not proof of fraud. A large family sharing
    one card and device would also form a component. The `known_fraud_rate`
    field below lets you see how predictive component membership actually
    is, rather than assuming it.
    """
    import networkx as nx

    adjacency = pd.read_parquet(GRAPH_OUT / f"{split_name}_edges.parquet")
    meta = pd.read_parquet(GRAPH_OUT / f"{split_name}_node_meta.parquet")

    G = nx.Graph()
    G.add_nodes_from(range(len(meta)))
    G.add_edges_from(adjacency[["src_idx", "dst_idx"]].itertuples(index=False, name=None))

    rings = []
    for component in nx.connected_components(G):
        if len(component) < min_ring_size:
            continue
        idxs = sorted(component)
        component_meta = meta.iloc[idxs]
        rings.append({
            "ring_id": f"RING-{len(rings) + 1:04d}",
            "size": len(idxs),
            "transaction_ids": component_meta["TransactionID"].tolist()[:50],
            "n_distinct_proxy_uids": int(component_meta["proxy_uid"].nunique()),
            "known_fraud_count": int(component_meta["isFraud"].sum()),
            "known_fraud_rate": round(float(component_meta["isFraud"].mean()), 4),
        })

    rings.sort(key=lambda r: (-r["known_fraud_rate"], -r["size"]))
    with open(GRAPH_OUT / f"{split_name}_fraud_rings.json", "w") as f:
        json.dump(rings, f, indent=2)
    return rings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the IEEE-CIS entity graph (no PyTorch required)")
    parser.add_argument("--sample-size", type=int, default=50000,
                        help="rows per split (default 50000; use 0 for the full split)")
    parser.add_argument("--min-shared-entities", type=int, default=2)
    parser.add_argument("--max-entity-degree", type=int, default=500)
    args = parser.parse_args()

    sample = None if args.sample_size == 0 else args.sample_size

    for split in ["train", "val", "test"]:
        stats = build_graph_from_split(
            split, sample_size=sample,
            min_shared_entities=args.min_shared_entities,
            max_entity_degree=args.max_entity_degree,
        )
        print(f"{split}: {json.dumps(stats)}")
        rings = extract_fraud_rings(split)
        print(f"  -> {len(rings)} candidate rings (>=3 transactions)")
