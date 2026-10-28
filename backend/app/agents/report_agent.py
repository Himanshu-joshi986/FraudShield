"""
Agent 4 -- Report Architect (Section 21).

Generates the structured investigation report FROM THE EVIDENCE DOSSIER
ONLY -- it must not invent information beyond what Evidence Analyst
compiled. Explicitly labeled as a demonstration draft, never as an
official regulatory filing (Section 21's explicit requirement).

NO LLM CALL required for the structural report (deterministic template
over the evidence list, guaranteeing every section is evidence-backed by
construction). An optional LLM narrative paragraph is attempted and
falls back gracefully, same pattern as Investigation Agent.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from app.agents.state import FraudCaseState
from app.agents.llm_client import get_llm_client


def _build_report_content(state: FraudCaseState) -> str:
    case_id = state["case_id"]
    case = state["input_data"]
    evidence = state.get("evidence", [])
    fusion = state.get("model_results", {}).get("fusion", {})
    adjustment = fusion.get("risk_adjustment") or {}
    if adjustment.get("type") == "evidence_floor":
        signal_labels = {
            "duplicate_invoice_across_cases": "Duplicate-invoice evidence",
            "claim_amount_mismatch": "Invoice amount-mismatch evidence",
            "high_transaction_velocity": "High-velocity evidence",
        }
        signal_label = signal_labels.get(adjustment.get("signal"), "Review evidence")
        floor_level = "HIGH" if adjustment.get("floor", 0) >= 0.65 else "MEDIUM"
        adjustment_section = (
            f"   Underlying signal average: {adjustment['base_score']:.2f}.\n"
            + (
                f"   {signal_label} raised the final assessment to "
                f"{adjustment['floor']:.2f} ({floor_level}).\n"
                if adjustment.get("applied") else
                f"   The underlying average already meets or exceeds the {floor_level} floor; "
                f"{signal_label.lower()} independently requires review.\n"
            )
            + "   This finding warrants review; it does not by itself establish fraud.\n"
        )
    else:
        adjustment_section = ""

    by_type: dict[str, list] = {}
    for e in evidence:
        by_type.setdefault(e["type"], []).append(e)

    def section(title, etype):
        items = by_type.get(etype, [])
        if not items:
            return f"{title}: no findings of this type for this case.\n"
        lines = "\n".join(f"  [{e['evidence_id']}] {e['finding']}" for e in items)
        return f"{title}:\n{lines}\n"

    return f"""FraudShield Investigation Report
(Demonstration draft -- NOT an official regulatory filing)

Case ID: {case_id}
Domain: {case.get('domain', 'unknown')}
Amount: ${case.get('amount', 0):,.2f}
Generated: {datetime.now(timezone.utc).isoformat()}

1. EXECUTIVE SUMMARY
   Risk score: {state.get('risk_score', 0):.2f} ({state.get('risk_level', 'UNSCORED')})
   Signals: {', '.join(state.get('signals', [])) or 'none'}
   Available signals used in scoring: {fusion.get('available_signals', [])}
   Signals not available for this case: {fusion.get('missing_signals', [])}
{adjustment_section}

2. GRAPH ANALYSIS
{section('   Graph findings', 'graph')}
3. DOCUMENT ANALYSIS
{section('   Document findings', 'document')}
4. REGULATORY / POLICY REFERENCES
{section('   Regulatory context', 'regulatory')}
5. MODEL EXPLANATION
{section('   Model explanation', 'model_explanation')}
6. UNCERTAINTY
   Risk scores and graph connectivity are indicators requiring review,
   not proof of fraud. This report should not be treated as an
   accusation. See knowledge base document KB-005 for the full
   responsible-use principles this system follows.

7. RECOMMENDED DISPOSITION
   Determined by Case Commander per the current routing policy (see the
   routing_decision recorded separately for this case).
"""


def run(state: FraudCaseState) -> FraudCaseState:
    t0 = time.time()

    content = _build_report_content(state)

    # optional LLM-generated narrative paragraph, degrades gracefully
    llm = get_llm_client()
    narrative_prompt = (
        f"Write one neutral, factual paragraph (3-4 sentences) summarizing this fraud "
        f"investigation report for a compliance officer. Use only the facts given, no "
        f"speculation, no accusatory language:\n\n{content[:1500]}"
    )
    llm_result = llm.complete(narrative_prompt)
    narrative = llm_result.text if llm_result.succeeded else (
        "Narrative summary unavailable (LLM synthesis degraded in this run); "
        "see the structured sections above for the full evidence-backed findings."
    )

    report = {
        "report_type": "investigation_report_draft",
        "content": content,
        "narrative_summary": narrative,
        "narrative_generated_by": "gemini" if llm_result.succeeded else "unavailable",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Demonstration draft. Not an official regulatory filing.",
    }
    state["report"] = report

    log_entry = {
        "agent_name": "Report Architect",
        "status": "completed",
        "input_summary": f"case {state['case_id']}, {len(state.get('evidence', []))} evidence items",
        "output_summary": f"generated {len(content)}-char report "
                          f"(narrative: {'gemini' if llm_result.succeeded else 'template fallback'})",
        "output_json": {"report_type": report["report_type"],
                        "narrative_generated_by": report["narrative_generated_by"]},
        "duration_ms": round((time.time() - t0) * 1000, 2),
    }
    state.setdefault("agent_logs", []).append(log_entry)
    return state
