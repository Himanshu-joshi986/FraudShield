"""
Phase 9 -- Shared agent state (Section 23).

Every agent reads/writes only the fields it needs. This is the state
object threaded through the LangGraph StateGraph in graph_pipeline.py.
"""
from __future__ import annotations

from typing import TypedDict, Any


class FraudCaseState(TypedDict, total=False):
    case_id: str
    input_data: dict                     # raw case record (customer_id, amount, timestamp, domain, ...)

    model_results: dict                  # ML-layer outputs (tabular / graph / temporal / document risk)
    risk_score: float
    risk_level: str                      # LOW | MEDIUM | HIGH
    signals: list[str]                   # named risk indicators, e.g. "high_transaction_velocity"

    graph_findings: dict                 # Investigation Agent's Cypher query results
    investigation_findings: dict         # observed_fact / inference / external_knowledge / uncertainty
    retrieved_knowledge: list[dict]      # RAG results with provenance

    evidence: list[dict]                 # structured evidence dossier (Section 20 shape)
    explanations: dict                   # SHAP-style contribution breakdown (Phase 10)

    report: dict                         # {report_type, content, ...}
    routing_decision: dict               # {decision, reason, policy_conditions}

    agent_logs: list[dict]               # per-agent execution log (mirrors AgentExecution table)
    errors: list[dict]                   # non-fatal errors, so one failing agent doesn't crash the run

    short_circuit: bool                  # Sentinel can end the pipeline early for low-risk, unconnected cases
