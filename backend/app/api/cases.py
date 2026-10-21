import uuid
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.session import get_db
from app.database import models
from app.schemas import case as schemas
from app.graph.store import get_graph_store
from app.rag.knowledge_base import get_knowledge_base
from app.services.pipeline_service import investigate_case

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

    graph_store = get_graph_store()
    knowledge_base = get_knowledge_base()
    try:
        result = await investigate_case(db, case_row, graph_store, knowledge_base)
    finally:
        graph_store.close()

    return result


@router.get("/{case_id}/graph", response_model=schemas.CaseGraphOut)
def get_case_graph(case_id: str, db: Session = Depends(get_db)):
    """Return the case's connected-entity graph for the case-detail tab."""
    case_row = db.query(models.Case).filter(models.Case.case_id == case_id).first()
    if not case_row:
        raise HTTPException(404, f"Case {case_id} not found")
    if not case_row.customer_id:
        return schemas.CaseGraphOut(case_id=case_id, nodes=[], edges=[])

    graph_store = get_graph_store()
    try:
        sub = graph_store.get_customer_subgraph(case_row.customer_id)
    finally:
        graph_store.close()
    return schemas.CaseGraphOut(case_id=case_id, nodes=sub["nodes"], edges=sub["edges"])


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
