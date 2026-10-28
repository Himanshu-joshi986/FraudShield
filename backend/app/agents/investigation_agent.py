"""
Agent 2 -- Investigation (Section 19).

Tool calls (graph query, RAG query) are REAL and run in this
environment. The LLM synthesis step (turning raw tool output into a
structured investigation narrative) calls Gemini; since that's
unreachable from this sandbox (verified: 403 Forbidden), it degrades to
a deterministic template that still produces the same structured output
shape -- observed_fact / inference / external_knowledge / uncertainty,
per Section 19's explicit requirement that these be distinguished.

Section 19: "Do not allow the LLM to present unsupported speculation as
fact." The deterministic fallback enforces this by construction (it only
ever states what the tools actually returned); the LLM path is
instructed to do the same via the prompt template.
"""
from __future__ import annotations

import time

from app.agents.state import FraudCaseState
from app.agents.llm_client import get_llm_client
from app.graph.store import GraphStore
from app.rag.knowledge_base import KnowledgeBase

RAG_QUERY_MAP = {
    "fraud_ring_graph_connectivity": "coordinated fraud ring shared account device indicators",
    "shared_entity_detected": "shared infrastructure between claimants indicators",
    "claim_amount_mismatch": "suspicious activity report dollar threshold filing",
    "duplicate_invoice_across_cases": "duplicate invoice reused across claims",
    "high_transaction_velocity": "claim velocity spike rapid filing",
}


def _build_prompt(case_id: str, graph_findings: dict, retrieved_knowledge: list[dict]) -> str:
    connections_text = "\n".join(
        f"  - shares {c['shared_type']} with customer {c['connected_customer']}"
        for c in graph_findings.get("connections", [])
    ) or "  - none found"
    kb_text = "\n".join(
        f"  - [{r['document_id']}] {r['snippet'][:150]}" for r in retrieved_knowledge
    ) or "  - none retrieved"

    return f"""You are the Investigation Agent in a fraud-investigation system.
Case: {case_id}

Graph connections found (from a real database query, not inference):
{connections_text}

Regulatory knowledge base passages retrieved:
{kb_text}

Write a brief investigation finding. You MUST separate your output into
these four sections, clearly labeled:
- OBSERVED FACT: only what the tool outputs above literally show.
- INFERENCE: reasonable conclusions drawn from those facts, labeled as inference.
- EXTERNAL KNOWLEDGE: what the retrieved regulatory passages say.
- UNCERTAINTY: what remains unknown or would need further verification.
Do not state anything as fact that isn't directly supported by the tool output above."""


def _deterministic_fallback(case_id: str, graph_findings: dict, retrieved_knowledge: list[dict]) -> dict:
    """
    Used when the LLM is unavailable (as it is in this build environment).
    Produces the same four-part structure using only tool output --
    this is, if anything, a STRICTER guarantee against unsupported
    speculation than an LLM-generated narrative would be.
    """
    connections = graph_findings.get("connections", [])
    n = len({c["connected_customer"] for c in connections})

    observed_fact = (
        f"Graph query found {n} other customer(s) connected to this case via shared "
        f"account/device/address." if n else
        "Graph query found no other customers connected to this case via shared entities."
    )
    inference = (
        f"The presence of {n} connected customers is consistent with coordinated activity "
        f"and warrants further review." if n >= 3 else
        "No inference of coordination is warranted from graph connectivity alone." if n == 0 else
        f"{n} connection(s) found; this alone is weak evidence and should be combined with "
        f"other signals before drawing conclusions."
    )
    external_knowledge = (
        "; ".join(f"[{r['document_id']}] {r['title']}" for r in retrieved_knowledge)
        or "No directly relevant regulatory passages were retrieved for this case."
    )
    uncertainty = (
        "This finding is generated from a deterministic template (LLM synthesis unavailable in "
        "this run) and reflects only the literal tool output above; a human investigator should "
        "verify connected identities independently before any determination."
    )

    return {
        "observed_fact": observed_fact, "inference": inference,
        "external_knowledge": external_knowledge, "uncertainty": uncertainty,
        "generated_by": "deterministic_fallback",
    }


def run(state: FraudCaseState, graph_store: GraphStore, knowledge_base: KnowledgeBase) -> FraudCaseState:
    t0 = time.time()
    case_id = state["case_id"]
    customer_id = state["input_data"].get("customer_id")

    # --- REAL tool call: graph query ---
    connections = graph_store.find_connected_customers(customer_id) if customer_id else []
    graph_findings = {"connections": connections, "n_connections": len(connections)}

    # --- REAL tool call: RAG query, driven by Sentinel's named signals ---
    queries = [RAG_QUERY_MAP[s] for s in state.get("signals", []) if s in RAG_QUERY_MAP]
    if not queries:
        queries = ["fraud investigation risk indicators"]
    retrieved = []
    for q in queries:
        retrieved.extend(knowledge_base.query(q, top_k=2, case_id=case_id))
    seen = set()
    deduped = []
    for r in retrieved:
        key = (r["document_id"], r["snippet"][:60])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # --- LLM synthesis: try Gemini, fall back deterministically ---
    llm = get_llm_client()
    prompt = _build_prompt(case_id, graph_findings, deduped)
    llm_result = llm.complete(prompt)

    if llm_result.succeeded:
        investigation_findings = {"narrative": llm_result.text, "generated_by": "gemini"}
        degraded = False
    else:
        investigation_findings = _deterministic_fallback(case_id, graph_findings, deduped)
        degraded = True

    state["graph_findings"] = graph_findings
    state["retrieved_knowledge"] = deduped
    state["investigation_findings"] = investigation_findings

    log_entry = {
        "agent_name": "Investigation Agent",
        "status": "completed",
        "input_summary": f"case {case_id}",
        "output_summary": (
            f"graph: {graph_findings['n_connections']} connection(s) found; "
            f"RAG: {len(deduped)} passage(s) retrieved; "
            f"LLM synthesis: {'gemini' if not degraded else 'degraded (' + str(llm_result.degraded_reason) + ')'}"
        ),
        "output_json": {"graph_findings": graph_findings, "retrieved_knowledge": deduped,
                        "investigation_findings": investigation_findings},
        "duration_ms": round((time.time() - t0) * 1000, 2),
    }
    if degraded:
        state.setdefault("errors", []).append({
            "agent": "Investigation Agent", "component": "llm_synthesis",
            "reason": llm_result.degraded_reason, "action_taken": "used deterministic fallback",
        })
    state.setdefault("agent_logs", []).append(log_entry)
    return state
