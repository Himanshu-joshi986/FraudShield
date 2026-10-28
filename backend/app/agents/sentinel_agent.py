"""
Agent 1 -- Sentinel (Section 18).

Monitors incoming cases, collects whatever risk signals are actually
available for this case (graph connectivity, document validation,
velocity -- tabular ML when the case carries the right feature shape,
see risk_fusion.py's docstring), and produces the initial risk_score /
risk_level / recommended_next_step.

NO LLM CALL. Sentinel's job is pure signal collection and fusion --
deterministic, auditable, and fast, which is also why it runs first and
can short-circuit obviously-clean cases before any expensive downstream
work (including the rate-limited LLM calls in Investigation/Report).
"""
from __future__ import annotations

import time

from app.agents.state import FraudCaseState
from app.ml.fusion.risk_fusion import fuse, RISK_LEVEL_THRESHOLDS
from app.graph.store import GraphStore, OVERRIDE_ELIGIBLE_TYPES
from app.documents import validator as doc_validator
from app.services.velocity import as_utc, event_signature

SHORT_CIRCUIT_THRESHOLD = 0.15  # below this AND zero graph connections, skip the rest of the pipeline

# Velocity heuristic for case-level records with no IEEE-CIS feature vector.
# A burst means at least three distinct customer events within a trailing
# window. Future events and same-time ties cannot count as prior activity.
VELOCITY_WINDOW_SECONDS = 3600
VELOCITY_HIGH_COUNT = 3
DUPLICATE_INVOICE_RISK_FLOOR = RISK_LEVEL_THRESHOLDS["MEDIUM"]
HUMAN_REVIEW_RISK_FLOOR = RISK_LEVEL_THRESHOLDS["LOW"]


def _velocity_risk(case: dict, all_cases: list[dict] | None,
                   case_id: str | None = None) -> tuple[float | None, list[str], int | None]:
    if all_cases is None or not case.get("customer_id"):
        return None, [], None
    this_time = as_utc(case.get("timestamp"))
    if this_time is None:
        return None, [], None
    window_start = this_time.timestamp() - VELOCITY_WINDOW_SECONDS
    prior_events = set()
    for item in all_cases:
        if item.get("customer_id") != case["customer_id"] or item.get("case_id") == case_id:
            continue
        event_time = as_utc(item.get("timestamp"))
        # Strictly earlier: ordering between separate records with the same
        # timestamp is unknown. Future activity must never affect this case.
        if event_time is None or not (window_start <= event_time.timestamp() < this_time.timestamp()):
            continue
        prior_events.add(event_signature(item.get("customer_id"), event_time,
                                         item.get("amount"), item.get("domain")))

    burst_count = len(prior_events) + 1  # count this case exactly once
    # Preserve the existing 9-event calibration while removing the baseline
    # event: one event has no velocity risk; the full 9-event burst reaches 1.
    risk = min(1.0, max(0, burst_count - 1) / 8.0)
    flags = ["high_transaction_velocity"] if burst_count >= VELOCITY_HIGH_COUNT else []
    return risk, flags, burst_count


def run(state: FraudCaseState, graph_store: GraphStore, document_path: str | None = None,
        all_cases: list[dict] | None = None, duplicate_invoice_case_ids: set | None = None) -> FraudCaseState:
    t0 = time.time()
    case = state["input_data"]
    customer_id = case.get("customer_id")

    signals: dict[str, float | None] = {
        "tabular_ml_risk": None,   # not available for synthetic scenario cases -- see risk_fusion.py
        "temporal_risk": None,
        "graph_risk": None,
        "document_risk": None,
    }
    named_signals = []
    n_override_eligible_connections = 0

    # --- graph signal (real Cypher query against the Phase-7 store) ---
    if customer_id:
        connections = graph_store.find_connected_customers(customer_id)
        connected_ids = {c["connected_customer"] for c in connections}

        override_connections = {
            c["connected_customer"] for c in connections if c["shared_type"] in OVERRIDE_ELIGIBLE_TYPES
        }
        n_override_eligible_connections = len(override_connections)
        weak_only_connections = connected_ids - override_connections

        # weak-only connections (e.g. address-sharing alone, no account/
        # device overlap) contribute to graph_risk, but at reduced weight --
        # otherwise a purely address-based cluster can still push the raw
        # fused score into HIGH even though it was correctly excluded from
        # the hard override. Found during Phase 9 evaluation (Scenario 6).
        graph_risk = min(1.0, n_override_eligible_connections / 5.0) \
                   + min(0.3, len(weak_only_connections) * 0.1)
        signals["graph_risk"] = min(1.0, graph_risk)

        if connected_ids:
            named_signals.append("shared_entity_detected")
        if n_override_eligible_connections >= 3:
            named_signals.append("fraud_ring_graph_connectivity")

    # --- velocity signal (real, computed from case timestamps) ---
    velocity_risk, velocity_flags, velocity_burst_count = _velocity_risk(case, all_cases, state["case_id"])
    signals["temporal_risk"] = velocity_risk
    named_signals.extend(velocity_flags)
    state["velocity_burst_count"] = velocity_burst_count
    state["velocity_window_seconds"] = VELOCITY_WINDOW_SECONDS

    # --- document signal (real OCR + validation against the Phase-6 layer) ---
    if document_path:
        findings = doc_validator.validate_document(
            image_path=document_path,
            claim_amount=case.get("amount", 0.0),
            claimant_name=case.get("customer_name", ""),
            case_id=state["case_id"],
        )
        signals["document_risk"] = findings.document_risk
        named_signals.extend(findings.flags)

    # cross-case duplicate-invoice flag, precomputed once for the whole
    # batch (a single document's own validation can't see OTHER cases --
    # this was a real gap found during Phase 9 evaluation, see
    # docs/EXPERIMENTS.md §11)
    if duplicate_invoice_case_ids and state["case_id"] in duplicate_invoice_case_ids:
        if "duplicate_invoice_across_cases" not in named_signals:
            named_signals.append("duplicate_invoice_across_cases")
        signals["document_risk"] = max(signals["document_risk"] or 0.0, 0.6)

    fusion_result = fuse(signals)
    base_risk_score = fusion_result["risk_score"]
    duplicate_invoice_override = "duplicate_invoice_across_cases" in named_signals
    review_signal = next((signal for signal in (
        "claim_amount_mismatch", "high_transaction_velocity",
    ) if signal in named_signals), None)
    # Duplicate claims carry a stronger HIGH minimum. Other deterministic
    # review overrides establish MEDIUM as a floor so the displayed band does
    # not contradict the human-review routing decision.
    adjustment_signal = "duplicate_invoice_across_cases" if duplicate_invoice_override else review_signal
    adjustment_floor = DUPLICATE_INVOICE_RISK_FLOOR if duplicate_invoice_override else HUMAN_REVIEW_RISK_FLOOR
    if adjustment_signal and fusion_result["available_signals"]:
        fusion_result["base_risk_score"] = base_risk_score
        fusion_result["risk_adjustment"] = {
            "type": "evidence_floor",
            "signal": adjustment_signal,
            "base_score": base_risk_score,
            "floor": adjustment_floor,
            "applied": base_risk_score < adjustment_floor,
            "reason": {
                "duplicate_invoice_across_cases": "same invoice number was detected on multiple separate cases",
                "claim_amount_mismatch": "validated invoice amount does not match the claim amount",
                "high_transaction_velocity": "multiple distinct customer events occurred within the review window",
            }[adjustment_signal],
        }
        fusion_result["risk_score"] = max(base_risk_score, adjustment_floor)
        fusion_result["risk_level"] = "HIGH" if fusion_result["risk_score"] >= RISK_LEVEL_THRESHOLDS["MEDIUM"] else (
            "MEDIUM" if fusion_result["risk_score"] >= RISK_LEVEL_THRESHOLDS["LOW"] else "LOW"
        )

    state["model_results"] = {"signals": signals, "fusion": fusion_result}
    state["risk_score"] = fusion_result["risk_score"]
    state["risk_level"] = fusion_result["risk_level"]
    state["signals"] = named_signals
    state["n_override_eligible_connections"] = n_override_eligible_connections
    state["short_circuit"] = (
        fusion_result["risk_score"] < SHORT_CIRCUIT_THRESHOLD
        and bool(fusion_result["available_signals"])
        and (signals.get("graph_risk") or 0) == 0
        and not named_signals
    )

    recommended = "MONITOR" if state["short_circuit"] else "INVESTIGATE"

    log_entry = {
        "agent_name": "Sentinel Agent",
        "status": "completed",
        "input_summary": f"case {state['case_id']}, customer {customer_id}",
        "output_summary": (
            f"risk_score={fusion_result['risk_score']:.2f} ({fusion_result['risk_level']}), "
            f"signals={named_signals or 'none'}, recommended_next_step={recommended}"
        ),
        "output_json": {
            "case_id": state["case_id"], "risk_score": fusion_result["risk_score"],
            "risk_level": fusion_result["risk_level"], "signals": named_signals,
            # The per-signal VALUES (not just their names) are recorded so
            # the scorecard in the dashboard can decompose the fused score
            # into the exact contributions that produced it. "signals"
            # above stays the list of named flags, for compatibility.
            "signal_values": signals,
            "velocity_burst_count": velocity_burst_count,
            "velocity_window_seconds": VELOCITY_WINDOW_SECONDS,
            "recommended_next_step": recommended, "fusion_detail": fusion_result,
        },
        "duration_ms": round((time.time() - t0) * 1000, 2),
    }
    state.setdefault("agent_logs", []).append(log_entry)
    return state
