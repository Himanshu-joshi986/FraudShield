import uuid
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.session import get_db
from app.database import models
from app.schemas import case as schemas
from app.graph.store import graph_session
from app.rag.knowledge_base import get_knowledge_base
from app.services.pipeline_service import investigate_case
from app.services.case_insights import build_case_insights, build_document_dossier

router = APIRouter(prefix="/api/cases", tags=["cases"])


def _next_case_id(db: Session) -> str:
    count = db.query(func.count(models.Case.case_id)).scalar() or 0
    return f"FS-{count + 1:05d}"


@router.post("", response_model=schemas.CaseOut)
def create_case(payload: schemas.CaseCreate, db: Session = Depends(get_db)):
    case_id = _next_case_id(db)
    row = models.Case(
        case_id=case_id,
        source=payload.source,
        domain=payload.domain,
        customer_id=payload.customer_id,
        customer_name=payload.customer_name,
        transaction_id=payload.transaction_id,
        claim_id=payload.claim_id,
        amount=payload.amount,
        timestamp=payload.timestamp or datetime.now(timezone.utc),
        data_source=payload.data_source,
        scenario_id=payload.scenario_id,
        risk_level="UNSCORED",
        status="new",
    )
    if payload.transaction_id and not db.query(models.Transaction).filter(
        models.Transaction.transaction_id == payload.transaction_id
    ).first():
        db.add(models.Transaction(
            transaction_id=payload.transaction_id,
            customer_id=payload.customer_id,
            amount=payload.amount,
            timestamp=payload.timestamp or row.timestamp,
            data_source=payload.data_source,
            scenario_id=payload.scenario_id,
            raw_features={"entry_type": "case_intake"},
        ))
        db.flush()
    if payload.claim_id and not db.query(models.Claim).filter(
        models.Claim.claim_id == payload.claim_id
    ).first():
        db.add(models.Claim(
            claim_id=payload.claim_id,
            customer_id=payload.customer_id,
            amount=payload.amount,
            date=payload.timestamp or row.timestamp,
            data_source=payload.data_source,
            scenario_id=payload.scenario_id,
            raw_features={"entry_type": "case_intake"},
        ))
        db.flush()
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("", response_model=list[schemas.CaseOut])
def list_cases(
    status: str | None = None,
    risk_level: str | None = None,
    data_source: str | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    q = db.query(models.Case)
    if status:
        q = q.filter(models.Case.status == status)
    if risk_level:
        q = q.filter(models.Case.risk_level == risk_level)
    if data_source:
        q = q.filter(models.Case.data_source == data_source)
    return q.order_by(models.Case.created_at.desc()).limit(limit).all()


@router.delete("/{case_id}", status_code=204)
def delete_case(case_id: str, db: Session = Depends(get_db)):
    """Remove a case and its saved investigation artifacts.

    Linked source transaction/claim rows and supporting documents are retained:
    they are source material and may be shared or needed if the case is filed
    again. Generated case reports are removed when they are in the reports
    directory and match this case's generated report filename.
    """
    row = db.query(models.Case).filter(models.Case.case_id == case_id).first()
    if not row:
        raise HTTPException(404, f"Case {case_id} not found")

    report_paths = [report.file_path for report in row.reports if report.file_path]
    db.delete(row)  # Case relationships cascade to evidence, reports, decisions, and agent logs.
    db.commit()

    reports_dir = (Path(__file__).resolve().parents[3] / "reports").resolve()
    expected_name = f"FraudShield_Report_{case_id}.pdf"
    for stored_path in report_paths:
        try:
            candidate = Path(stored_path).resolve()
            if candidate.parent == reports_dir and candidate.name == expected_name:
                candidate.unlink(missing_ok=True)
        except (OSError, RuntimeError, ValueError):
            # The database deletion should succeed even if an old generated
            # report file cannot be cleaned up.
            continue

    return Response(status_code=204)


@router.get("/{case_id}", response_model=schemas.CaseDetailOut)
def get_case(case_id: str, db: Session = Depends(get_db)):
    row = db.query(models.Case).filter(models.Case.case_id == case_id).first()
    if not row:
        raise HTTPException(404, f"Case {case_id} not found")
    return schemas.CaseDetailOut(
        case=row,
        agent_executions=sorted(row.agent_executions, key=lambda a: a.execution_id),
        evidence=row.evidence,
        reports=row.reports,
        decisions=row.decisions,
    )


@router.get("/{case_id}/evidence", response_model=list[schemas.EvidenceOut])
def get_case_evidence(case_id: str, db: Session = Depends(get_db)):
    row = db.query(models.Case).filter(models.Case.case_id == case_id).first()
    if not row:
        raise HTTPException(404, f"Case {case_id} not found")
    return row.evidence


@router.get("/{case_id}/report", response_model=list[schemas.ReportOut])
def get_case_report(case_id: str, db: Session = Depends(get_db)):
    row = db.query(models.Case).filter(models.Case.case_id == case_id).first()
    if not row:
        raise HTTPException(404, f"Case {case_id} not found")
    return row.reports


@router.get("/{case_id}/report/pdf")
def download_case_report_pdf(case_id: str, db: Session = Depends(get_db)):
    report = db.query(models.Report).filter(models.Report.case_id == case_id).first()
    if not report or not report.file_path or not Path(report.file_path).exists():
        raise HTTPException(404, f"No PDF report found for case {case_id}")
    return FileResponse(report.file_path, media_type="application/pdf",
                        filename=Path(report.file_path).name)


@router.post("/{case_id}/investigate", response_model=schemas.InvestigateResultOut)
async def investigate_case_endpoint(case_id: str, db: Session = Depends(get_db)):
    """
    Runs the real 5-agent LangGraph pipeline (Phase 9) on this case and
    persists AgentExecution / Evidence / Report / Decision rows, while
    broadcasting real agent_started/agent_completed WebSocket events
    (Phase 13, Section 40/41) to anyone subscribed via
    WS /ws/cases/{case_id}.
    """
    case_row = db.query(models.Case).filter(models.Case.case_id == case_id).first()
    if not case_row:
        raise HTTPException(404, f"Case {case_id} not found")

    knowledge_base = get_knowledge_base()
    # The graph store is opened by the pipeline itself, per agent, so the
    # embedded database's exclusive file lock is not held for the whole
    # investigation (which includes OCR and LLM calls).
    result = await investigate_case(db, case_row, None, knowledge_base)

    return result


@router.get("/{case_id}/graph", response_model=schemas.CaseGraphOut)
def get_case_graph(case_id: str, db: Session = Depends(get_db)):
    """Return the case's connected-entity graph for the case-detail tab."""
    case_row = db.query(models.Case).filter(models.Case.case_id == case_id).first()
    if not case_row:
        raise HTTPException(404, f"Case {case_id} not found")
    if not case_row.customer_id:
        return schemas.CaseGraphOut(case_id=case_id, nodes=[], edges=[])

    # graph_session serializes access to the embedded graph database and
    # releases its exclusive file lock as soon as this request is done.
    with graph_session() as graph_store:
        sub = graph_store.get_customer_subgraph(case_row.customer_id)
    return schemas.CaseGraphOut(case_id=case_id, nodes=sub["nodes"], edges=sub["edges"])


@router.get("/{case_id}/insights", response_model=schemas.CaseInsightsOut)
def get_case_insights(case_id: str, db: Session = Depends(get_db)):
    """
    The numbers behind the case: the signal scorecard (a decomposition of
    the fused risk score), the routing trace, exact counterfactuals, and
    the case activity timeline.

    Every field is derived from data the pipeline actually produced --
    Sentinel's recorded signal values and Case Commander's recorded
    policy conditions -- so the scorecard always adds back up to the
    case's own risk_score.
    """
    case_row = db.query(models.Case).filter(models.Case.case_id == case_id).first()
    if not case_row:
        raise HTTPException(404, f"Case {case_id} not found")
    return build_case_insights(db, case_row)


@router.get("/{case_id}/documents", response_model=schemas.CaseDocumentsOut)
def get_case_documents(case_id: str, db: Session = Depends(get_db)):
    """
    The case's document dossier: the real OCR fields read from the
    supporting document, plus the other documents that filed the same
    invoice number, so the dashboard can show them side by side.

    OCR is cached (see app/services/document_store.py), so this endpoint
    does not re-run tesseract on every request.
    """
    case_row = db.query(models.Case).filter(models.Case.case_id == case_id).first()
    if not case_row:
        raise HTTPException(404, f"Case {case_id} not found")
    return build_document_dossier(case_id)


@router.post("/seed-synthetic")
def seed_synthetic_cases(db: Session = Depends(get_db)):
    """
    Loads the Phase-2 synthetic scenario library (data/synthetic/synthetic_cases.csv)
    into the Cases table via the API, bridging the CSV-based scenario
    library into the live database for dashboard/demo use.
    """
    import pandas as pd
    csv_path = Path(__file__).resolve().parents[3] / "data" / "synthetic" / "synthetic_cases.csv"
    if not csv_path.exists():
        raise HTTPException(404, "synthetic_cases.csv not found -- run scripts/generate_synthetic_scenarios.py first")

    df = pd.read_csv(csv_path)
    n_created = 0
    for _, row in df.iterrows():
        if db.query(models.Case).filter(models.Case.case_id == row["case_id"]).first():
            continue
        db.add(models.Case(
            case_id=row["case_id"], source="synthetic", domain=row["domain"],
            customer_id=row["customer_id"], customer_name=row["customer_name"],
            amount=float(row["amount"]), timestamp=pd.to_datetime(row["timestamp"]),
            risk_level="UNSCORED", status="new", data_source=row["data_source"],
            scenario_id=row["scenario_id"],
        ))
        n_created += 1
    db.commit()
    return {"seeded": n_created, "already_existed": len(df) - n_created}
