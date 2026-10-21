"""
Phase 12 prep / integration -- bridges the in-memory LangGraph pipeline
(Phase 9, which wrote to JSON files for evaluation) into the actual
Phase-1 database schema, so a live API call persists real
AgentExecution / Evidence / Report / Decision rows, not just files.

This is genuinely new integration work: nothing built in Phases 9-11
previously touched the database at all.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.database import models
from app.graph.store import GraphStore
from app.rag.knowledge_base import KnowledgeBase
from app.agents.graph_pipeline import build_graph
from app.services.websocket_manager import run_case_streaming
from app.explainability.explain_case import explain_case
from app.services.report_generator import generate_report_pdf

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS_MANIFEST = REPO_ROOT / "data" / "documents" / "manifest.json"


def _load_document_paths() -> dict:
    if not DOCUMENTS_MANIFEST.exists():
        return {}
    with open(DOCUMENTS_MANIFEST) as f:
        manifest = json.load(f)
    paths = {}
    for entry in manifest:
        paths.setdefault(entry["case_id"], str(REPO_ROOT / entry["image_path"]))
    return paths


_duplicate_cache: set | None = None


def _load_duplicate_invoice_case_ids() -> set:
    """
    Cached at module level: this previously re-ran OCR (tesseract) on
    every document in the manifest on EVERY investigate call, which made
    each investigation take ~12 seconds -- a real performance bug found
    while testing the dashboard live. Documents don't change between
    calls in this environment, so computing this once and reusing it is
    correct, not just faster.
    """
    global _duplicate_cache
    if _duplicate_cache is not None:
        return _duplicate_cache

    if not DOCUMENTS_MANIFEST.exists():
        _duplicate_cache = set()
        return _duplicate_cache

    from app.documents.validator import validate_document, detect_duplicate_documents

    with open(DOCUMENTS_MANIFEST) as f:
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
    detect_duplicate_documents(findings_list)
    _duplicate_cache = {f.case_id for f in findings_list if "duplicate_invoice_across_cases" in f.flags}
    return _duplicate_cache


async def investigate_case(db: Session, case_row: models.Case, graph_store: GraphStore,
                           knowledge_base: KnowledgeBase) -> dict:
    """
    Runs the real 5-agent pipeline on one case via LangGraph's streaming
    API (broadcasting real agent_started/agent_completed WebSocket events
    as each agent actually executes -- see websocket_manager.py) and
    persists every agent's execution, the evidence dossier, the generated
    PDF report, and the routing decision to the database.
    """
    all_cases = [
        {"customer_id": c.customer_id, "timestamp": c.timestamp.isoformat() if c.timestamp else None}
        for c in db.query(models.Case).all()
    ]
    document_paths = _load_document_paths()
    duplicate_invoice_case_ids = _load_duplicate_invoice_case_ids()

    compiled_graph = build_graph(graph_store, knowledge_base, document_paths,
                                 all_cases=all_cases,
                                 duplicate_invoice_case_ids=duplicate_invoice_case_ids)

    input_data = {
        "customer_id": case_row.customer_id, "customer_name": case_row.customer_name,
        "amount": case_row.amount, "domain": case_row.domain,
        "timestamp": case_row.timestamp.isoformat() if case_row.timestamp else datetime.now(timezone.utc).isoformat(),
        "scenario_id": case_row.scenario_id,
    }

    final_state = await run_case_streaming(compiled_graph, case_row.case_id, input_data)

    # idempotent re-investigation: clear this case's prior run before
    # inserting new rows, otherwise re-running investigate on the same
    # case would violate the Decision/Report tables' case-derived
    # primary keys.
    db.query(models.AgentExecution).filter(models.AgentExecution.case_id == case_row.case_id).delete()
    db.query(models.Evidence).filter(models.Evidence.case_id == case_row.case_id).delete()
    db.query(models.Report).filter(models.Report.case_id == case_row.case_id).delete()
    db.query(models.Decision).filter(models.Decision.case_id == case_row.case_id).delete()

    for log in final_state.get("agent_logs", []):
        db.add(models.AgentExecution(
            case_id=case_row.case_id, agent_name=log["agent_name"], status=log["status"],
            input_summary=log.get("input_summary", ""), output_summary=log.get("output_summary", ""),
            output_json=log.get("output_json", {}),
            started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc),
            duration_ms=log.get("duration_ms"),
        ))

    for item in final_state.get("evidence", []):
        db.add(models.Evidence(
            evidence_id=item["evidence_id"], case_id=case_row.case_id, type=item["type"],
            finding=item["finding"], source=item["source"],
            source_reference=item.get("source_reference", {}), confidence=item["confidence"],
        ))

    report_id = None
    if final_state.get("report"):
        explanation = explain_case(final_state)
        pdf_path = generate_report_pdf(final_state, explanation)
        report_id = f"R-{case_row.case_id}"
        db.add(models.Report(
            report_id=report_id, case_id=case_row.case_id,
            content=final_state["report"]["content"], file_path=str(pdf_path),
        ))

    routing = final_state.get("routing_decision", {})
    decision_id = f"D-{case_row.case_id}"
    db.add(models.Decision(
        decision_id=decision_id, case_id=case_row.case_id,
        decision=routing.get("decision", "UNKNOWN"), reason=routing.get("reason", ""),
        policy_version=routing.get("policy_version", "v1"),
    ))

    case_row.risk_score = final_state.get("risk_score", 0.0)
    case_row.risk_level = final_state.get("risk_level", "UNSCORED")
    case_row.status = routing.get("decision", "new")
    case_row.updated_at = datetime.now(timezone.utc)

    db.commit()

    return {
        "case_id": case_row.case_id, "status": case_row.status,
        "risk_score": case_row.risk_score, "risk_level": case_row.risk_level,
        "routing_decision": routing, "n_agents_run": len(final_state.get("agent_logs", [])),
        "n_evidence_items": len(final_state.get("evidence", [])),
        "n_errors": len(final_state.get("errors", [])),
        "short_circuited": bool(final_state.get("short_circuit")),
        "report_id": report_id,
    }
