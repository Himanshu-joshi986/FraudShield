"""GNNExplainer adapter for trained FraudSAGE checkpoints.

GNNExplainer answers a different question than SHAP: not "which raw
features drove this prediction" but "which EDGES and NEIGHBOUR NODES in
the graph drove this prediction" -- i.e. which specific shared-entity
relationships caused a transaction to be flagged.
"""
from __future__ import annotations


def explain_node(model, data, node_idx: int, epochs: int = 200) -> dict:
    """
    model: a trained FraudSAGE (see ml/gnn/model.py)
    data: the PyG Data object the model was trained/evaluated on
    node_idx: index of the transaction node to explain

    Returns a dict with the top contributing edges (as (src, dst) node
    index pairs with an importance score) and top contributing node
    features -- the two things GNNExplainer natively produces.
    """
    import torch
    from torch_geometric.explain import Explainer, GNNExplainer

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )

    explanation = explainer(data.x, data.edge_index, index=node_idx)

    edge_mask = explanation.edge_mask
    node_mask = explanation.node_mask

    top_edge_idx = torch.argsort(edge_mask, descending=True)[:10]
    top_edges = [
        {
            "source": int(data.edge_index[0, i]), "target": int(data.edge_index[1, i]),
            "importance": round(float(edge_mask[i]), 4),
        }
        for i in top_edge_idx if edge_mask[i] > 0
    ]

    feature_importance = None
    if node_mask is not None:
        mask = node_mask.detach().cpu()
        feature_importance = (
            mask[node_idx].numpy() if mask.ndim > 1 and mask.shape[0] > node_idx
            else mask.reshape(-1).numpy()
        )
    top_features = []
    if feature_importance is not None:
        import numpy as np
        ranked = np.argsort(-np.abs(feature_importance))[:5]
        top_features = [
            {"feature_index": int(i), "importance": round(float(feature_importance[i]), 4)}
            for i in ranked
        ]

    return {
        "node_idx": node_idx,
        "top_contributing_edges": top_edges,
        "top_contributing_features": top_features,
        "generated_by": "GNNExplainer (torch_geometric.explain)",
    }


def explain_and_map_to_transactions(model, data, node_meta, node_idx: int) -> dict:
    """
    Wraps explain_node() and maps the raw node indices back to
    TransactionIDs (using the node_meta parquet from
    ml/gnn/graph_builder.py), so the explanation is directly usable by
    the Evidence Analyst / dashboard rather than needing a separate
    lookup step.
    """
    raw = explain_node(model, data, node_idx)
    raw["transaction_id"] = str(node_meta.iloc[node_idx]["TransactionID"])
    for edge in raw["top_contributing_edges"]:
        edge["source_transaction_id"] = str(node_meta.iloc[edge["source"]]["TransactionID"])
        edge["target_transaction_id"] = str(node_meta.iloc[edge["target"]]["TransactionID"])
    return raw
