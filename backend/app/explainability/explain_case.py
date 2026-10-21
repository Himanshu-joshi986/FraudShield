"""
Phase 10c -- "Explain This Case" (Section 51, signature feature).

Turns the evidence dossier Evidence Analyst compiled into the numbered,
clickable "WHY WAS THIS CASE FLAGGED?" explanation Section 51 specifies.
This is deliberately NOT a new inference step -- it formats and ranks
evidence that already exists with full provenance (Section 28's evidence
chain), so every numbered reason links back to a real evidence_id, which
links back to a real source (a Cypher query result, an OCR'd document, a
retrieved regulation).
"""
from __future__ import annotations


def explain_case(state: dict) -> dict:
    """
    state: a completed FraudCaseState (post Evidence Analyst).
    Returns the Section 51 structure: ranked reasons, each with its
    supporting evidence_id, confidence, and a pointer to the underlying
    source -- clickable in the dashboard because evidence_id resolves
    directly to a row in the Evidence table.
    """
    evidence = state.get("evidence", [])
    ranked = sorted(evidence, key=lambda e: e.get("confidence", 0.0), reverse=True)

    reasons = []
    for i, item in enumerate(ranked, start=1):
        reasons.append({
            "rank": i,
            "reason": item["finding"],
            "evidence_id": item["evidence_id"],
            "evidence_type": item["type"],
            "confidence": item["confidence"],
            "source": item["source"],
        })

    fusion = state.get("model_results", {}).get("fusion", {})

    return {
        "case_id": state.get("case_id"),
        "risk_score": state.get("risk_score"),
        "risk_level": state.get("risk_level"),
        "headline": f"Flagged with risk score {state.get('risk_score', 0):.2f} "
                    f"({state.get('risk_level', 'UNSCORED')}) based on {len(reasons)} evidence item(s).",
        "reasons": reasons,
        "signals_available": fusion.get("available_signals", []),
        "signals_not_applicable": fusion.get("missing_signals", []),
        "routing_decision": state.get("routing_decision"),
        "disclaimer": (
            "This explanation describes model and evidence signals. It does not "
            "establish that fraud actually occurred -- see knowledge base document "
            "KB-005 for the full responsible-use principles this system follows."
        ),
    }
