"""
Phase 10c -- verifies explain_case() against a REAL pipeline run.

Runs the actual 5-agent pipeline (same as scripts/run_agent_pipeline.py)
on the Scenario 3 fraud-ring case, then formats the result through
explain_case() -- confirming the full chain (Sentinel -> Investigation ->
Evidence -> explain_case) produces a real, evidence-backed explanation
end to end, not against mocked state.
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.agents.graph_pipeline import build_graph, run_case  # noqa: E402
from app.graph.store import get_graph_store  # noqa: E402
from app.rag.knowledge_base import get_knowledge_base  # noqa: E402
from app.explainability.explain_case import explain_case  # noqa: E402

SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
DOCUMENTS_DIR = REPO_ROOT / "data" / "documents"
OUT_PATH = REPO_ROOT / "docs" / "explain_case_example.json"


def main():
    cases_df = pd.read_csv(SYNTHETIC_DIR / "synthetic_cases.csv")

    with open(DOCUMENTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    document_paths = {}
    for entry in manifest:
        document_paths.setdefault(entry["case_id"], str(REPO_ROOT / entry["image_path"]))

    graph_store = get_graph_store()
    knowledge_base = get_knowledge_base()
    compiled_graph = build_graph(graph_store, knowledge_base, document_paths,
                                 all_cases=cases_df.to_dict("records"))

    target = cases_df[cases_df["scenario_id"] == "FS-SC-003"].iloc[0]
    final_state = run_case(compiled_graph, target["case_id"], {
        "customer_id": target["customer_id"], "customer_name": target["customer_name"],
        "amount": float(target["amount"]), "domain": target["domain"],
        "timestamp": target["timestamp"], "scenario_id": target["scenario_id"],
    })

    explanation = explain_case(final_state)

    with open(OUT_PATH, "w") as f:
        json.dump(explanation, f, indent=2)

    print(f"Case: {explanation['case_id']}")
    print(explanation["headline"])
    print()
    for r in explanation["reasons"]:
        print(f"  {r['rank']}. {r['reason'][:100]}")
        print(f"     [{r['evidence_id']}] confidence={r['confidence']:.2f} source={r['source']}")
    print(f"\nSaved -> {OUT_PATH}")

    graph_store.close()


if __name__ == "__main__":
    main()
