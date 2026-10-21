from pydantic import BaseModel, ConfigDict, Field, model_validator
from datetime import datetime
from typing import Literal, Optional, Any, List


class CaseCreate(BaseModel):
    source: Literal["ieee_cis", "carclaims", "synthetic", "manual"] = Field(
        ..., description="ieee_cis | carclaims | synthetic | manual"
    )
    domain: Literal["financial_transaction", "insurance_claim"]
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    transaction_id: Optional[str] = None
    claim_id: Optional[str] = None
    amount: float = Field(..., gt=0, allow_inf_nan=False)
    timestamp: Optional[datetime] = None
    data_source: str = "public_benchmark"
    scenario_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_domain_source(self):
        if self.source == "ieee_cis" and self.domain != "financial_transaction":
            raise ValueError("ieee_cis cases must use the financial_transaction domain")
        if self.source == "carclaims" and self.domain != "insurance_claim":
            raise ValueError("carclaims cases must use the insurance_claim domain")
        if self.domain == "financial_transaction" and self.claim_id:
            raise ValueError("financial_transaction cases cannot include claim_id")
        if self.domain == "insurance_claim" and self.transaction_id:
            raise ValueError("insurance_claim cases cannot include transaction_id")
        return self


class CaseOut(BaseModel):
    case_id: str
    source: str
    domain: str
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    transaction_id: Optional[str]
    claim_id: Optional[str]
    amount: float
    timestamp: datetime
    risk_score: float
    risk_level: str
    status: str
    data_source: str
    scenario_id: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AgentExecutionOut(BaseModel):
    execution_id: int
    agent_name: str
    status: str
    input_summary: Optional[str]
    output_summary: Optional[str]
    output_json: Any
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_ms: Optional[float]
    error: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class EvidenceOut(BaseModel):
    evidence_id: str
    type: str
    finding: str
    source: str
    source_reference: Any
    confidence: float
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ReportOut(BaseModel):
    report_id: str
    content: str
    file_path: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DecisionOut(BaseModel):
    decision_id: str
    decision: str
    reason: str
    policy_version: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CaseDetailOut(BaseModel):
    case: CaseOut
    agent_executions: List[AgentExecutionOut]
    evidence: List[EvidenceOut]
    reports: List[ReportOut]
    decisions: List[DecisionOut]


class HealthOut(BaseModel):
    status: str
    app_env: str
    db_connected: bool


class AnalyticsOut(BaseModel):
    total_cases: int
    auto_close_monitor: int
    human_review: int
    escalated: int
    unscored: int
    avg_risk_score: float
    known_fraud_scenarios_missed: int  # synthetic ground-truth check, see analytics endpoint


class GraphNodeOut(BaseModel):
    id: str
    type: str
    name: Optional[str] = None


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    relation: str


class CaseGraphOut(BaseModel):
    case_id: str
    nodes: List[GraphNodeOut]
    edges: List[GraphEdgeOut]


class InvestigateResultOut(BaseModel):
    case_id: str
    status: str
    risk_score: float
    risk_level: str
    routing_decision: dict
    n_agents_run: int
    n_evidence_items: int
    n_errors: int
    short_circuited: bool
    report_id: Optional[str] = None
