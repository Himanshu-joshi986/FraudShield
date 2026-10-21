"""
Agent 5 -- Case Commander (Section 22).

Routes the case per the policy in knowledge_base/regulatory/case_routing_policy.txt
(KB-003), including its two override rules. The specific policy
conditions that triggered the decision are always recorded (Section 22's
explicit requirement), so a routing decision is auditable after the
fact, not a black box.

NO LLM CALL. Routing is a policy application, not a generation task --
it must be deterministic and reproducible given the same risk score and
evidence.
"""
from __future__ import annotations

import time

from app.agents.state import FraudCaseState

HIGH_THRESHOLD = 0.65
MEDIUM_THRESHOLD = 0.35
RING_OVERRIDE_SIZE = 3
DUPLICATE_OVERRIDE_MIN_CLAIMS = 2


def run(state: FraudCaseState) -> FraudCaseState:
    t0 = time.time()
    risk_score = state.get("risk_score", 0.0)
    # Use the OVERRIDE-ELIGIBLE connection count Sentinel computed (Account/
    # Device only), NOT graph_findings.n_connections (Investigation Agent's
    # count, which includes ALL shared-entity types including Address).
    # These were two different code paths computing "ring size" -- a real
    # bug found during Phase 9 evaluation that caused Scenario 6
    # (address-only sharing) to incorrectly trigger the ring override.
    n_connections = state.get("n_override_eligible_connections", 0)
    signals = state.get("signals", [])
    fusion = state.get("model_results", {}).get("fusion", {})

    conditions_triggered = []
    no_available_signals = not fusion.get("available_signals") and fusion.get("degraded", False)

    # Override 1: coordinated-ring override (KB-003)
    ring_override = n_connections >= RING_OVERRIDE_SIZE
    if ring_override:
        conditions_triggered.append(
            f"ring_override: {n_connections} connected customers >= {RING_OVERRIDE_SIZE}"
        )

    # Override 2: duplicate-invoice override (KB-003)
    duplicate_override = "duplicate_invoice_across_cases" in signals
    if duplicate_override:
        conditions_triggered.append("duplicate_invoice_override: shared invoice number across claims")

    # Override 3: document amount-mismatch override (KB-003) -- added
    # during Phase 9 evaluation. A confirmed claim-vs-invoice amount
    # mismatch is a specific, well-defined finding (not a vague "low
    # score") and, per KB-004's red-flag guidance, warrants at least
    # human review on its own rather than being averaged away by an
    # otherwise-quiet case.
    document_mismatch_override = "claim_amount_mismatch" in signals
    if document_mismatch_override:
        conditions_triggered.append("document_mismatch_override: claim amount does not match invoice")

    # Override 4: velocity-burst override (KB-003) -- added during Phase 9
    # evaluation. A confirmed high-velocity burst (see
    # sentinel_agent.VELOCITY_HIGH_COUNT) is a specific, well-defined
    # finding, same reasoning as the document-mismatch override: it
    # shouldn't be averaged away by an otherwise-quiet case's other
    # signals sitting at zero.
    velocity_override = "high_transaction_velocity" in signals
    if velocity_override:
        conditions_triggered.append("velocity_override: high transaction velocity burst detected")

    if no_available_signals:
        decision = "HUMAN_REVIEW"
        conditions_triggered.append("no risk signals available; manual review required")
    elif ring_override or duplicate_override or risk_score >= HIGH_THRESHOLD:
        decision = "ESCALATE"
        if not conditions_triggered:
            conditions_triggered.append(f"risk_score {risk_score:.2f} >= HIGH threshold {HIGH_THRESHOLD}")
    elif risk_score >= MEDIUM_THRESHOLD or document_mismatch_override or velocity_override:
        decision = "HUMAN_REVIEW"
        if risk_score >= MEDIUM_THRESHOLD:
            conditions_triggered.append(f"risk_score {risk_score:.2f} in MEDIUM band [{MEDIUM_THRESHOLD}, {HIGH_THRESHOLD})")
    else:
        decision = "AUTO_CLOSE_MONITOR"
        conditions_triggered.append(f"risk_score {risk_score:.2f} < MEDIUM threshold {MEDIUM_THRESHOLD}")

    routing_decision = {
        "decision": decision,
        "reason": "; ".join(conditions_triggered),
        "policy_conditions_triggered": conditions_triggered,
        "policy_version": "v1",
        "policy_source": "KB-003 (case_routing_policy.txt)",
    }
    state["routing_decision"] = routing_decision

    log_entry = {
        "agent_name": "Case Commander",
        "status": "completed",
        "input_summary": f"case {state['case_id']}, risk_score={risk_score:.2f}",
        "output_summary": f"routed to {decision}. {routing_decision['reason']}",
        "output_json": routing_decision,
        "duration_ms": round((time.time() - t0) * 1000, 2),
    }
    state.setdefault("agent_logs", []).append(log_entry)
    return state
