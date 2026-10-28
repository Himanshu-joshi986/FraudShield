"""
Agent 3 -- Evidence Analyst (Section 20).

Compiles the structured evidence dossier from Sentinel + Investigation's
output. Every evidence item carries a traceable source_reference (Section
28's evidence-chain requirement) -- not just a finding string, but a
pointer back to the underlying record.

NO LLM CALL. This agent's job is structured compilation, not generation --
deliberately kept deterministic so the evidence dossier is exactly
reproducible from the same inputs, which matters for an investigation
artifact.

SHAP / GNNExplainer integration (Section 20's `tools` list) is built out
fully in Phase 10; this agent already emits a `model_explanation`
evidence item using the fusion weights as a placeholder contribution
breakdown, upgraded in Phase 10 without changing this agent's structure.
"""
from __future__ import annotations

import time

from app.agents.state import FraudCaseState
from app.services.document_store import findings_for_case

_evidence_counter = {"n": 0}


def _next_evidence_id(case_id: str) -> str:
    _evidence_counter["n"] += 1
    return f"E-{case_id}-{_evidence_counter['n']:03d}"


def run(state: FraudCaseState) -> FraudCaseState:
    t0 = time.time()
    case_id = state["case_id"]
    evidence = []

    graph_findings = state.get("graph_findings", {})
    if graph_findings.get("n_connections", 0) > 0:
        evidence.append({
            "evidence_id": _next_evidence_id(case_id),
            "type": "graph",
            "finding": f"{graph_findings['n_connections']} other customer(s) connected via shared entities.",
            "source": "graph_analytics (kuzu Cypher query)",
            "source_reference": {"connections": graph_findings["connections"]},
            "confidence": min(1.0, graph_findings["n_connections"] / 5.0),
        })

    doc_signal = state.get("model_results", {}).get("signals", {}).get("document_risk")
    if doc_signal is not None:
        evidence.append({
            "evidence_id": _next_evidence_id(case_id),
            "type": "document",
            "finding": f"Document validation risk score: {doc_signal:.2f}. Flags: {state.get('signals', [])}.",
            "source": "document_ai (OCR + cross-verification)",
            "source_reference": {"document_risk": doc_signal, "flags": state.get("signals", [])},
            "confidence": doc_signal,
        })

    # Preserve the actual duplicate-invoice observation in the dossier,
    # including the invoice and all affected case IDs. A generic policy
    # flag alone is not enough for a reviewer to verify the finding.
    document_finding = findings_for_case(case_id)
    if document_finding and "duplicate_invoice_across_cases" in document_finding.flags:
        invoice = document_finding.extracted.get("primary_invoice_number")
        shared_cases = sorted(set(
            document_finding.extracted.get("duplicate_shared_with_cases", []) + [case_id]
        ))
        evidence.append({
            "evidence_id": _next_evidence_id(case_id),
            "type": "document",
            "finding": (
                f"Invoice {invoice or 'number unreadable'} was detected on separate cases: "
                f"{', '.join(shared_cases)}. This duplicate-filing pattern requires review; "
                "it is not by itself a final finding of fraud."
            ),
            "source": "document_ai (OCR cross-case invoice comparison)",
            "source_reference": {
                "invoice_number": invoice,
                "case_ids": shared_cases,
                "image_path": document_finding.image_path,
                "ocr_backend": document_finding.ocr_backend,
            },
            "confidence": document_finding.ocr_confidence or 0.0,
        })

    for kb_result in state.get("retrieved_knowledge", []):
        evidence.append({
            "evidence_id": _next_evidence_id(case_id),
            "type": "regulatory",
            "finding": f"[{kb_result['title']}] {kb_result['snippet'][:200]}",
            "source": "rag_knowledge_base (ChromaDB)",
            "source_reference": {"document_id": kb_result["document_id"],
                                "section": kb_result["section"]},
            "confidence": max(0.0, 1.0 - kb_result["distance"] / 2.0),
        })

    inv_findings = state.get("investigation_findings", {})
    if inv_findings:
        evidence.append({
            "evidence_id": _next_evidence_id(case_id),
            "type": "model_explanation",
            "finding": inv_findings.get("narrative") or inv_findings.get("observed_fact", ""),
            "source": f"investigation_agent ({inv_findings.get('generated_by', 'unknown')})",
            "source_reference": {"generated_by": inv_findings.get("generated_by")},
            "confidence": 0.5 if inv_findings.get("generated_by") == "deterministic_fallback" else 0.7,
        })

    # contribution breakdown from the fusion weights actually used.
    # Note on scope: SHAP explainability (Phase 10, backend/app/
    # explainability/shap_explainer.py) is real and RUNS against the
    # Phase-3 tabular models for cases that carry the full IEEE-CIS
    # feature vector. Synthetic scenario cases (like this one) don't
    # carry that feature vector -- see risk_fusion.py's architectural
    # note -- so their model_explanation is this fusion-weight
    # breakdown instead, which is the correct and complete explanation
    # for what actually drove THIS case's score.
    fusion = state.get("model_results", {}).get("fusion", {})
    signal_values = state.get("model_results", {}).get("signals", {})
    no_signals = fusion.get("degraded") and not fusion.get("available_signals")
    evidence.append({
        "evidence_id": _next_evidence_id(case_id),
        "type": "model_explanation",
        "finding": (
            f"Risk score unavailable: no usable risk signals were available. "
            f"Missing signals: {fusion.get('missing_signals', [])}; manual review is required."
            if no_signals else
            f"Risk score {state.get('risk_score', 0):.2f} derived from signals: "
            f"{fusion.get('available_signals', [])} (weights: {fusion.get('weights_used', {})})."
        ),
        "source": "risk_fusion (Sentinel Agent) -- SHAP available separately for cases "
                 "carrying a full tabular feature vector, see docs/EXPERIMENTS.md §13",
        # The per-signal values live here as well as in Sentinel's own
        # execution record, so the scorecard can always be rebuilt from
        # the evidence dossier alone.
        "source_reference": {**fusion, "signal_values": signal_values},
        "confidence": 0.0 if fusion.get("degraded") else 1.0,
    })

    state["evidence"] = evidence

    log_entry = {
        "agent_name": "Evidence Analyst",
        "status": "completed",
        "input_summary": f"case {case_id}",
        "output_summary": f"compiled {len(evidence)} evidence item(s): "
                          f"{[e['type'] for e in evidence]}",
        "output_json": {"evidence": evidence},
        "duration_ms": round((time.time() - t0) * 1000, 2),
    }
    state.setdefault("agent_logs", []).append(log_entry)
    return state
