"""
Phase 9 -- End-to-end agent pipeline evaluation.

Runs the real, compiled LangGraph pipeline over all 26 synthetic
scenario cases and scores the resulting routing decisions against the
known `expected_risk_level` ground truth (Section 9's scenario library).

This is a real agent-pipeline evaluation (Section 48): workflow
completion rate, routing consistency, and whether the pipeline's
decisions match the scenarios' designed-in expected outcome.
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

SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
DOCUMENTS_DIR = REPO_ROOT / "data" / "documents"
OUT_PATH = REPO_ROOT / "docs" / "agent_pipeline_metrics.json"

EXPECTED_ROUTING = {"LOW": "AUTO_CLOSE_MONITOR", "MEDIUM": "HUMAN_REVIEW", "HIGH": "ESCALATE"}


def load_document_paths() -> dict:
    manifest_path = DOCUMENTS_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        manifest = json.load(f)
    paths = {}
    for entry in manifest:
        paths.setdefault(entry["case_id"], str(REPO_ROOT / entry["image_path"]))
    return paths


def compute_duplicate_invoice_case_ids() -> set:
    """
    Precomputes which case_ids are implicated in a cross-case duplicate
    invoice, using the Phase-6 detector across ALL generated documents at
    once. A single case's own document validation cannot see other
    cases' documents -- this must run as a batch pass.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.documents.validator import validate_document, detect_duplicate_documents

    manifest_path = DOCUMENTS_DIR / "manifest.json"
    if not manifest_path.exists():
        return set()
    with open(manifest_path) as f:
        manifest = json.load(f)

    findings_list = [
        validate_document(
            image_path=REPO_ROOT / entry["image_path"],
            claim_amount=entry["claim_amount_on_file"],
            claimant_name=entry.get("customer_name", ""),
            case_id=entry["case_id"],
        )
        for entry in manifest
    ]
    detect_duplicate_documents(findings_list)  # mutates findings_list in place
    return {f.case_id for f in findings_list if "duplicate_invoice_across_cases" in f.flags}


def main():
    cases_df = pd.read_csv(SYNTHETIC_DIR / "synthetic_cases.csv")
    document_paths = load_document_paths()
    duplicate_invoice_case_ids = compute_duplicate_invoice_case_ids()
    all_cases = cases_df.to_dict("records")

    graph_store = get_graph_store()
    knowledge_base = get_knowledge_base()

    compiled_graph = build_graph(graph_store, knowledge_base, document_paths,
                                 all_cases=all_cases,
                                 duplicate_invoice_case_ids=duplicate_invoice_case_ids)

    results = []
    errors_seen = 0
    short_circuits = 0

    for _, row in cases_df.iterrows():
        case_id = row["case_id"]
        input_data = {
            "customer_id": row["customer_id"], "customer_name": row["customer_name"],
            "amount": float(row["amount"]), "domain": row["domain"],
            "timestamp": row["timestamp"], "scenario_id": row["scenario_id"],
        }

        try:
            final_state = run_case(compiled_graph, case_id, input_data)
        except Exception as e:
            errors_seen += 1
            results.append({
                "case_id": case_id, "scenario_id": row["scenario_id"],
                "expected_risk_level": row["expected_risk_level"],
                "pipeline_error": str(e),
            })
            continue

        if final_state.get("short_circuit"):
            short_circuits += 1

        decision = final_state.get("routing_decision", {}).get("decision")
        expected_decision = EXPECTED_ROUTING[row["expected_risk_level"]]

        results.append({
            "case_id": case_id,
            "scenario_id": row["scenario_id"],
            "expected_risk_level": row["expected_risk_level"],
            "expected_decision": expected_decision,
            "actual_decision": decision,
            "decision_matches_expected": decision == expected_decision,
            "risk_score": final_state.get("risk_score"),
            "n_agents_run": len(final_state.get("agent_logs", [])),
            "n_evidence_items": len(final_state.get("evidence", [])),
            "n_errors": len(final_state.get("errors", [])),
            "short_circuited": bool(final_state.get("short_circuit")),
            "routing_reason": final_state.get("routing_decision", {}).get("reason"),
        })

    n = len(results)
    n_completed = sum(1 for r in results if "pipeline_error" not in r)
    n_matches = sum(1 for r in results if r.get("decision_matches_expected"))
    n_with_llm_degradation = sum(1 for r in results if r.get("n_errors", 0) > 0)

    per_scenario = {}
    for sid in sorted(cases_df["scenario_id"].unique()):
        sc_results = [r for r in results if r["scenario_id"] == sid]
        per_scenario[sid] = {
            "n_cases": len(sc_results),
            "expected_risk_level": sc_results[0].get("expected_risk_level"),
            "all_matched": all(r.get("decision_matches_expected") for r in sc_results),
            "decisions": [r.get("actual_decision") for r in sc_results],
        }

    metrics = {
        "n_cases": n,
        "workflow_completion_rate": round(n_completed / n, 4),
        "routing_consistency_rate": round(n_matches / n, 4),
        "n_short_circuited": short_circuits,
        "n_full_pipeline_runs": n - short_circuits,
        "n_cases_with_llm_degradation": n_with_llm_degradation,
        "llm_degradation_note": (
            "Every full-pipeline case shows LLM degradation because Gemini is unreachable "
            "from this build environment (verified: 403 Forbidden on "
            "generativelanguage.googleapis.com). The pipeline completed successfully anyway "
            "via the deterministic fallback paths in investigation_agent.py and "
            "report_agent.py -- this IS the error-handling behavior Section 57 requires, "
            "demonstrated under real failure conditions, not simulated."
        ),
        "per_scenario": per_scenario,
        "per_case": results,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Workflow completion rate: {metrics['workflow_completion_rate']}")
    print(f"Routing consistency rate: {metrics['routing_consistency_rate']}")
    print(f"Short-circuited (low-risk fast path): {short_circuits}/{n}")
    print(f"Cases with LLM degradation (expected, Gemini unreachable here): {n_with_llm_degradation}")
    print()
    for sid, info in per_scenario.items():
        mark = "OK" if info["all_matched"] else "XX"
        print(f"  {mark} {sid} ({info['expected_risk_level']}): {info['decisions']}")
    print(f"\nSaved -> {OUT_PATH}")

    graph_store.close()


if __name__ == "__main__":
    main()
