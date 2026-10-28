"""
Phase 9 -- LangGraph orchestration (Section 24).

Real conditional StateGraph, not a linear chain of 5 LLM calls (Section
4's explicit anti-pattern). Sentinel runs first and can route
low-risk, unconnected cases straight to END -- avoiding rate-limited LLM
calls for cases with no signal at all -- while flagged cases go through
the full Investigation -> Evidence -> Report -> Commander chain.

    START -> Sentinel -> [risk gate]
                            |-- short_circuit=True  --> END (monitor)
                            |-- short_circuit=False --> Investigation
                                                          -> Evidence
                                                          -> Report
                                                          -> Commander -> END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agents.state import FraudCaseState
from app.agents import sentinel_agent, investigation_agent, evidence_agent, report_agent, case_commander
from app.graph.store import GraphStore, graph_session
from app.rag.knowledge_base import KnowledgeBase


def build_graph(graph_store: GraphStore | None, knowledge_base: KnowledgeBase, document_paths: dict[str, str],
                all_cases: list[dict] | None = None, duplicate_invoice_case_ids: set | None = None):
    """
    `graph_store`: an already-open store, or None.

    When None, each agent that needs the graph opens one for exactly as
    long as its own graph work takes. The embedded kuzu database is a
    single-writer resource (it takes an exclusive file lock), so holding
    that lock for a whole 5-agent investigation -- which includes OCR and
    LLM calls that have nothing to do with the graph -- would lock out
    every other reader of the graph for the duration. Scripts that
    already hold a store pass it in and it is used as-is.
    """
    """
    `document_paths`: {case_id: image_path} for cases that have a
    document to validate (Scenarios 4/5/7/8). Cases without an entry
    simply skip the document signal (Sentinel handles this as an
    unavailable signal, not an error).

    `all_cases`: full list of case dicts (customer_id, timestamp, ...)
    used for the velocity/burst signal -- a case must see its neighbors'
    timestamps to detect a filing burst, which a single case record alone
    cannot do.

    `duplicate_invoice_case_ids`: precomputed set of case_ids implicated
    in a cross-case duplicate invoice (Phase 6's batch detector) -- a
    single document's own OCR/validation pass has no visibility into
    OTHER cases' documents, so this must be computed once for the whole
    batch and passed in.
    """

    def _with_store(fn):
        """Runs an agent with a graph store, opening one only if needed."""
        if graph_store is not None:
            return fn(graph_store)
        with graph_session() as store:
            return fn(store)

    def sentinel_node(state: FraudCaseState) -> FraudCaseState:
        doc_path = document_paths.get(state["case_id"])

        def run(store):
            return sentinel_agent.run(state, store, document_path=doc_path,
                                      all_cases=all_cases,
                                      duplicate_invoice_case_ids=duplicate_invoice_case_ids)

        return _with_store(run)

    def investigation_node(state: FraudCaseState) -> FraudCaseState:
        def run(store):
            return investigation_agent.run(state, store, knowledge_base)

        return _with_store(run)

    def evidence_node(state: FraudCaseState) -> FraudCaseState:
        return evidence_agent.run(state)

    def report_node(state: FraudCaseState) -> FraudCaseState:
        return report_agent.run(state)

    def commander_node(state: FraudCaseState) -> FraudCaseState:
        return case_commander.run(state)

    def route_after_sentinel(state: FraudCaseState) -> str:
        if state.get("short_circuit"):
            return "short_circuit_end"
        return "investigate"

    def short_circuit_end_node(state: FraudCaseState) -> FraudCaseState:
        """Low-risk cases still get a routing_decision recorded (auto-close
        is itself an auditable decision, not silence)."""
        state["routing_decision"] = {
            "decision": "AUTO_CLOSE_MONITOR",
            "reason": f"Sentinel short-circuit: risk_score {state.get('risk_score', 0):.2f} "
                     f"below threshold, zero graph connections.",
            "policy_conditions_triggered": ["sentinel_short_circuit"],
            "policy_version": "v1",
            "policy_source": "sentinel_agent.SHORT_CIRCUIT_THRESHOLD",
        }
        return state

    graph = StateGraph(FraudCaseState)
    graph.add_node("sentinel", sentinel_node)
    graph.add_node("short_circuit_end", short_circuit_end_node)
    graph.add_node("investigation", investigation_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("report", report_node)
    graph.add_node("commander", commander_node)

    graph.set_entry_point("sentinel")
    graph.add_conditional_edges("sentinel", route_after_sentinel, {
        "short_circuit_end": "short_circuit_end",
        "investigate": "investigation",
    })
    graph.add_edge("short_circuit_end", END)
    graph.add_edge("investigation", "evidence")
    graph.add_edge("evidence", "report")
    graph.add_edge("report", "commander")
    graph.add_edge("commander", END)

    return graph.compile()


def run_case(compiled_graph, case_id: str, input_data: dict) -> FraudCaseState:
    initial_state: FraudCaseState = {
        "case_id": case_id, "input_data": input_data,
        "agent_logs": [], "errors": [],
    }
    return compiled_graph.invoke(initial_state)
