"""
Phase 7 -- Graph ingestion + fraud-ring query evaluation.

Loads the Phase-2 synthetic scenario library into the graph store, then
runs real Cypher fraud-ring queries against it and scores them against
the KNOWN expected outcome of each scenario.

This is the first phase where the scenario library earns its keep: it has
ground-truth `expected_risk_level` per case, so we can measure whether
graph queries actually surface the rings they're supposed to (Scenario 3:
a 6-entity ring; Scenario 6: 4 claimants sharing an address) without
over-flagging the legitimate cases (Scenario 1, Scenario 7).
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.graph.store import get_graph_store, STRONG_SHARED_TYPES, WEAK_SHARED_TYPES  # noqa: E402

SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
OUT_PATH = REPO_ROOT / "docs" / "graph_metrics.json"

# maps the relation names in synthetic_relationships.csv to graph entity types
RELATION_TO_ENTITY = {
    "owns_account": ("Account", "OWNS_ACCOUNT"),
    "uses_device": ("Device", "USES_DEVICE"),
    "has_email": ("Email", "HAS_EMAIL"),
    "has_address": ("Address", "HAS_ADDRESS"),
    "filed_via_broker": ("Broker", "FILED_VIA"),
}


def ingest(store):
    cases = pd.read_csv(SYNTHETIC_DIR / "synthetic_cases.csv")
    relationships = pd.read_csv(SYNTHETIC_DIR / "synthetic_relationships.csv")

    store.initialize_schema(reset=True)

    for _, row in cases.drop_duplicates("customer_id").iterrows():
        store.add_customer(
            customer_id=row["customer_id"], name=row["customer_name"],
            data_source=row["data_source"], scenario_id=row["scenario_id"],
        )

    n_links = 0
    for _, row in relationships.iterrows():
        relation = row["relation"]
        if relation not in RELATION_TO_ENTITY:
            continue
        entity_type, rel_name = RELATION_TO_ENTITY[relation]
        store.add_entity(entity_type, str(row["target_entity"]))
        store.link(rel_name, "Customer", str(row["source_entity"]),
                   entity_type, str(row["target_entity"]))
        n_links += 1

    return {"customers": int(cases["customer_id"].nunique()), "links": n_links}


def evaluate(store):
    cases = pd.read_csv(SYNTHETIC_DIR / "synthetic_cases.csv")

    # strong-signal clusters (account/device/address shared by 2+ customers)
    clusters = {}
    for etype in STRONG_SHARED_TYPES:
        found = store.find_shared_entity_clusters(etype, min_customers=2)
        if found:
            clusters[etype] = found

    # weak-signal clusters reported separately -- broker/email sharing is
    # common among legitimate claimants and must NOT be treated as equal
    # evidence (see the note in graph/store.py)
    weak_clusters = {}
    for etype in WEAK_SHARED_TYPES:
        found = store.find_shared_entity_clusters(etype, min_customers=2)
        if found:
            weak_clusters[etype] = found

    # per-customer: how many others are they connected to via STRONG entities?
    per_customer = []
    for _, row in cases.drop_duplicates("customer_id").iterrows():
        cid = row["customer_id"]
        connections = store.find_connected_customers(cid)
        connected_ids = {c["connected_customer"] for c in connections}
        per_customer.append({
            "customer_id": cid,
            "scenario_id": row["scenario_id"],
            "expected_risk_level": row["expected_risk_level"],
            "n_strong_connections": len(connected_ids),
            "shared_types": sorted({c["shared_type"] for c in connections}),
        })

    df = pd.DataFrame(per_customer)

    # Does graph connectivity separate HIGH-risk scenarios from LOW?
    by_risk = df.groupby("expected_risk_level")["n_strong_connections"].agg(["count", "mean", "max"])

    # Scenario-level check: did we detect the rings we planted?
    scenario_checks = {}
    for sid in sorted(df["scenario_id"].unique()):
        subset = df[df["scenario_id"] == sid]
        scenario_checks[sid] = {
            "n_customers": int(len(subset)),
            "expected_risk_level": subset["expected_risk_level"].iloc[0],
            "max_strong_connections": int(subset["n_strong_connections"].max()),
            "mean_strong_connections": round(float(subset["n_strong_connections"].mean()), 2),
            "any_customer_connected": bool((subset["n_strong_connections"] > 0).any()),
        }

    return {
        "strong_signal_clusters": {
            etype: [{"entity": c["entity_id"], "n_customers": int(c["n_customers"])}
                    for c in found]
            for etype, found in clusters.items()
        },
        "weak_signal_clusters_reported_separately": {
            etype: [{"entity": c["entity_id"], "n_customers": int(c["n_customers"])}
                    for c in found]
            for etype, found in weak_clusters.items()
        },
        "connectivity_by_expected_risk": {
            level: {"n_customers": int(r["count"]),
                    "mean_strong_connections": round(float(r["mean"]), 2),
                    "max_strong_connections": int(r["max"])}
            for level, r in by_risk.iterrows()
        },
        "per_scenario": scenario_checks,
    }


def main():
    store = get_graph_store()
    print(f"Graph backend: {store.name}")

    ingest_stats = ingest(store)
    print(f"Ingested: {ingest_stats}")

    results = evaluate(store)
    results["backend"] = store.name
    results["ingest"] = ingest_stats
    results["note_strong_vs_weak"] = (
        "Account/Device/Address sharing is treated as STRONG fraud-ring signal. "
        "Broker/Email sharing is reported SEPARATELY as weak signal, because many "
        "legitimate claimants share a broker or email domain -- conflating them "
        "produces a flood of false positives."
    )

    # example subgraph for the dashboard (Scenario 3 = the planted 6-entity ring)
    cases = pd.read_csv(SYNTHETIC_DIR / "synthetic_cases.csv")
    ring_customer = cases[cases["scenario_id"] == "FS-SC-003"]["customer_id"].iloc[0]
    results["example_subgraph_FS-SC-003"] = store.get_customer_subgraph(ring_customer)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    printable = {k: v for k, v in results.items() if k != "example_subgraph_FS-SC-003"}
    print(json.dumps(printable, indent=2)[:3000])
    sub = results["example_subgraph_FS-SC-003"]
    print(f"\nExample subgraph ({ring_customer}): "
          f"{len(sub['nodes'])} nodes, {len(sub['edges'])} edges")
    print(f"Saved -> {OUT_PATH}")

    store.close()


if __name__ == "__main__":
    main()
