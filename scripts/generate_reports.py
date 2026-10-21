"""
Phase 11 -- generates PDF reports for representative real cases and
verifies their content by extracting text back out (pdfplumber) and
checking that key facts actually appear in the rendered PDF -- not just
that a file was written.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.agents.graph_pipeline import build_graph, run_case  # noqa: E402
from app.graph.store import get_graph_store  # noqa: E402
from app.rag.knowledge_base import get_knowledge_base  # noqa: E402
from app.explainability.explain_case import explain_case  # noqa: E402
from app.services.report_generator import generate_report_pdf  # noqa: E402

SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
DOCUMENTS_DIR = REPO_ROOT / "data" / "documents"
OUT_PATH = REPO_ROOT / "docs" / "report_generation_metrics.json"

REPRESENTATIVE_SCENARIOS = ["FS-SC-001", "FS-SC-003", "FS-SC-004", "FS-SC-005", "FS-SC-006", "FS-SC-008"]


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

    results = []
    for sid in REPRESENTATIVE_SCENARIOS:
        row = cases_df[cases_df["scenario_id"] == sid].iloc[0]
        final_state = run_case(compiled_graph, row["case_id"], {
            "customer_id": row["customer_id"], "customer_name": row["customer_name"],
            "amount": float(row["amount"]), "domain": row["domain"],
            "timestamp": row["timestamp"], "scenario_id": row["scenario_id"],
        })
        explanation = explain_case(final_state)
        pdf_path = generate_report_pdf(final_state, explanation)

        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            extracted_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

        checks = {
            "case_id_appears": row["case_id"] in extracted_text,
            "risk_level_appears": final_state.get("risk_level", "") in extracted_text,
            "routing_decision_appears": final_state.get("routing_decision", {}).get("decision", "") in extracted_text,
            # only require an evidence_id to appear when there IS evidence --
            # a short-circuited low-risk case (Scenario 1) legitimately has
            # zero evidence items, and the check should not fail on that.
            "at_least_one_evidence_id_appears": (
                any(e["evidence_id"] in extracted_text for e in final_state.get("evidence", []))
                if final_state.get("evidence") else True
            ),
            "disclaimer_appears": "NOT an official regulatory filing" in extracted_text,
        }

        results.append({
            "case_id": row["case_id"], "scenario_id": sid,
            "pdf_path": str(pdf_path.relative_to(REPO_ROOT)),
            "pdf_size_bytes": pdf_path.stat().st_size,
            "n_pages": n_pages,
            "n_evidence_items": len(final_state.get("evidence", [])),
            "extracted_text_length": len(extracted_text),
            "content_verification": checks,
            "all_checks_passed": all(checks.values()),
        })
        print(f"{sid}: {pdf_path.name} ({n_pages}p, {pdf_path.stat().st_size} bytes) "
              f"verification={'PASS' if all(checks.values()) else 'FAIL'}")

    metrics = {
        "n_reports_generated": len(results),
        "n_verification_passed": sum(1 for r in results if r["all_checks_passed"]),
        "reports": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")

    graph_store.close()


if __name__ == "__main__":
    main()
