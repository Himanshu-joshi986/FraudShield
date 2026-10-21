"""
Analytics endpoint (Section 38). Computed live from the actual database
-- no hardcoded/fake statistics (Section 65's build rule).
"""
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.database import models
from app.schemas import case as schemas

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("", response_model=schemas.AnalyticsOut)
def get_analytics(db: Session = Depends(get_db)):
    cases = db.query(models.Case).all()
    total = len(cases)

    auto_close = sum(1 for c in cases if c.status == "AUTO_CLOSE_MONITOR")
    human_review = sum(1 for c in cases if c.status == "HUMAN_REVIEW")
    escalated = sum(1 for c in cases if c.status == "ESCALATE")
    unscored = sum(1 for c in cases if c.status == "new")

    scored = [c for c in cases if c.risk_level != "UNSCORED"]
    avg_risk = round(sum(c.risk_score for c in scored) / len(scored), 4) if scored else 0.0

    # Synthetic-scenario ground-truth check (Section 38: "if insufficient
    # historical data exists, clearly label analytics as demonstration" --
    # this specific number IS demonstration-only, computed against the
    # Phase-2 scenario library's expected_risk_level, not real-world outcomes)
    csv_path = Path(__file__).resolve().parents[3] / "data" / "synthetic" / "synthetic_cases.csv"
    missed = 0
    if csv_path.exists():
        expected = pd.read_csv(csv_path).set_index("case_id")["expected_risk_level"].to_dict()
        expected_routing = {"LOW": "AUTO_CLOSE_MONITOR", "MEDIUM": "HUMAN_REVIEW", "HIGH": "ESCALATE"}
        for c in cases:
            if c.case_id in expected and c.risk_level != "UNSCORED":
                exp_decision = expected_routing.get(expected[c.case_id])
                if exp_decision and c.status != exp_decision:
                    missed += 1

    return schemas.AnalyticsOut(
        total_cases=total, auto_close_monitor=auto_close, human_review=human_review,
        escalated=escalated, unscored=unscored, avg_risk_score=avg_risk,
        known_fraud_scenarios_missed=missed,
    )
