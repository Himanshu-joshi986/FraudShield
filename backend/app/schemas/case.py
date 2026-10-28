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


# ---------------------------------------------------------------------------
# Case insights (Signal Scorecard / policy trace / counterfactuals / timeline)
# ---------------------------------------------------------------------------

class SignalScorecardOut(BaseModel):
    name: str
    label: str
    value: Optional[float] = None
    baseline_value: Optional[float] = None  # display-only reference; never a model score
    available: bool
    weight: float
    weighted: Optional[float] = None      # value * weight, before normalisation
    contribution: Optional[float] = None  # share of the underlying signal average
    contribution_pct: Optional[float] = None  # that share, as a percentage
    weight_share_pct: Optional[float] = None  # share of the available weight pool
    note: Optional[str] = None


class CounterfactualOut(BaseModel):
    signal: str
    label: str
    target_label: str
    target_threshold: float
    current_value: float
    required_value: Optional[float] = None
    achievable: bool
    explanation: str


class PolicyTraceOut(BaseModel):
    decision: Optional[str] = None
    reason: Optional[str] = None
    policy_version: Optional[str] = None
    policy_source: Optional[str] = None
    conditions_triggered: List[str] = []
    overrides: List[str] = []
    score_driven: bool = True


class CaseEventOut(BaseModel):
    timestamp: Optional[datetime] = None
    kind: str
    title: str
    detail: Optional[str] = None
    case_id: Optional[str] = None
    risk_level: Optional[str] = None
    is_current: bool = False
    in_velocity_window: bool = False
    flagged: bool = False


class AgentRunOut(BaseModel):
    agent_name: str
    status: str
    duration_ms: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    summary: Optional[str] = None


class CaseInsightsOut(BaseModel):
    case_id: str
    investigated: bool
    risk_score: float
    base_risk_score: Optional[float] = None
    risk_level: str
    thresholds: dict
    signals: List[SignalScorecardOut] = []
    fusion: dict = {}
    policy: Optional[PolicyTraceOut] = None
    counterfactuals: List[CounterfactualOut] = []
    score_below_medium: bool = False
    events: List[CaseEventOut] = []
    agents: List[AgentRunOut] = []
    velocity_window_seconds: int = 0
    velocity_cluster_size: int = 0
    sibling_case_count: int = 0


# ---------------------------------------------------------------------------
# Document dossier (side-by-side invoice comparison)
# ---------------------------------------------------------------------------

class DocumentCellOut(BaseModel):
    value: Optional[str] = None
    sub: Optional[str] = None
    highlight: bool = False


class DocumentRowOut(BaseModel):
    label: str
    differs: bool = False
    cells: List[DocumentCellOut] = []


class DocumentOut(BaseModel):
    case_id: str
    customer_name: Optional[str] = None
    image_name: Optional[str] = None
    invoice_number: Optional[str] = None
    printed_amount: Optional[float] = None
    claim_amount_on_file: Optional[float] = None
    ocr_backend: Optional[str] = None
    ocr_confidence: Optional[float] = None
    document_risk: float = 0.0
    flags: List[str] = []
    is_current_case: bool = False
    degraded: bool = False
    degraded_reason: Optional[str] = None


class CaseDocumentsOut(BaseModel):
    case_id: str
    has_document: bool = False
    group_invoice_number: Optional[str] = None
    duplicate_across_cases: bool = False
    documents: List[DocumentOut] = []
    rows: List[DocumentRowOut] = []
    summary: Optional[str] = None
