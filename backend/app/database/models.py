"""
Core persistence schema, matching Section 39 of the FraudShield spec
exactly (cases, transactions, claims, agent_executions, evidence,
reports, decisions), plus provenance columns (`data_source`,
`scenario_id`) required by Section 8 so the dashboard can always
distinguish public-benchmark data from synthetic demonstration data.

Runs on SQLite by default (see core/config.py); swapping
FRAUDSHIELD_DB_URL to a postgresql:// URL moves this to Postgres with
zero model changes.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database.session import Base


def utcnow():
    return datetime.now(timezone.utc)


class Case(Base):
    __tablename__ = "cases"

    case_id = Column(String, primary_key=True, index=True)  # e.g. "FS-00042"
    source = Column(String)                # "ieee_cis" | "carclaims" | "synthetic"
    domain = Column(String)                # "financial_transaction" | "insurance_claim"
    customer_id = Column(String, nullable=True, index=True)
    customer_name = Column(String, nullable=True)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"), nullable=True)
    claim_id = Column(String, ForeignKey("claims.claim_id"), nullable=True)
    amount = Column(Float)
    timestamp = Column(DateTime, default=utcnow)
    risk_score = Column(Float, default=0.0)
    risk_level = Column(String, default="UNSCORED")  # LOW | MEDIUM | HIGH | UNSCORED
    status = Column(String, default="new")  # new | investigating | monitor | human_review | escalated
    data_source = Column(String, default="public_benchmark")  # "public_benchmark" | "synthetic"
    scenario_id = Column(String, nullable=True)  # e.g. "FS-SC-003" for synthetic cases
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    agent_executions = relationship("AgentExecution", back_populates="case", cascade="all, delete-orphan")
    evidence = relationship("Evidence", back_populates="case", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="case", cascade="all, delete-orphan")
    decisions = relationship("Decision", back_populates="case", cascade="all, delete-orphan")


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, nullable=True, index=True)
    amount = Column(Float)
    timestamp = Column(DateTime)
    device_id = Column(String, nullable=True, index=True)
    account_id = Column(String, nullable=True, index=True)
    merchant_id = Column(String, nullable=True)
    is_fraud = Column(Boolean, nullable=True)  # null = unlabeled (e.g. Kaggle test set)
    data_source = Column(String, default="ieee_cis")
    scenario_id = Column(String, nullable=True)
    raw_features = Column(JSON, default=dict)  # engineered feature snapshot for this txn


class Claim(Base):
    __tablename__ = "claims"

    claim_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, nullable=True, index=True)
    policy_id = Column(String, nullable=True)
    amount = Column(Float)
    date = Column(DateTime, nullable=True)
    fraud_label = Column(Boolean, nullable=True)
    data_source = Column(String, default="carclaims")
    scenario_id = Column(String, nullable=True)
    raw_features = Column(JSON, default=dict)


class AgentExecution(Base):
    __tablename__ = "agent_executions"

    execution_id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    agent_name = Column(String)
    status = Column(String, default="pending")  # pending | running | completed | failed | skipped
    input_summary = Column(Text, nullable=True)
    output_summary = Column(Text, nullable=True)
    output_json = Column(JSON, default=dict)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Float, nullable=True)
    error = Column(Text, nullable=True)

    case = relationship("Case", back_populates="agent_executions")


class Evidence(Base):
    __tablename__ = "evidence"

    evidence_id = Column(String, primary_key=True, index=True)  # e.g. "E001"
    case_id = Column(String, ForeignKey("cases.case_id"))
    type = Column(String)  # graph | temporal | document | transaction | regulatory | model_explanation
    finding = Column(Text)
    source = Column(String)              # e.g. "graph_analytics", "temporal_model", "rag_knowledge_base"
    source_reference = Column(JSON, default=dict)  # pointer back to the underlying record(s)
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utcnow)

    case = relationship("Case", back_populates="evidence")


class Report(Base):
    __tablename__ = "reports"

    report_id = Column(String, primary_key=True, index=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    content = Column(Text)
    file_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    case = relationship("Case", back_populates="reports")


class Decision(Base):
    __tablename__ = "decisions"

    decision_id = Column(String, primary_key=True, index=True)
    case_id = Column(String, ForeignKey("cases.case_id"))
    decision = Column(String)  # MONITOR | HUMAN_REVIEW | ESCALATE
    reason = Column(Text)
    policy_version = Column(String, default="v1")
    created_at = Column(DateTime, default=utcnow)

    case = relationship("Case", back_populates="decisions")
