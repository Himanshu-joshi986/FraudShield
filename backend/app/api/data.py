"""CSV intake for user supplied transaction and claim records."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import models
from app.database.session import get_db
from app.graph.store import get_graph_store

router = APIRouter(prefix="/api/data", tags=["data"])


class CsvImport(BaseModel):
    filename: str = "records.csv"
    domain: str = Field(pattern="^(financial_transaction|insurance_claim)$")
    csv_text: str = Field(min_length=1)


def _value(row: dict, *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


@router.post("/import-csv")
def import_csv(payload: CsvImport, db: Session = Depends(get_db)):
    """Import rows as persistent cases, source records, and shared graph entities."""
    try:
        reader = csv.DictReader(io.StringIO(payload.csv_text.lstrip("\ufeff")))
        if not reader.fieldnames or "amount" not in {h.strip().lower() for h in reader.fieldnames}:
            raise HTTPException(400, "CSV needs a header row with an amount column.")
        normalized = [{(k or "").strip().lower(): v for k, v in row.items()} for row in reader]
        if not normalized:
            raise HTTPException(400, "The CSV contains a header but no records.")
        if len(normalized) > 5000:
            raise HTTPException(413, "Import up to 5,000 rows at a time.")
    except csv.Error as exc:
        raise HTTPException(400, f"Could not read CSV: {exc}") from exc

    is_claim = payload.domain == "insurance_claim"
    created: list[models.Case] = []
    graph_rows: list[dict] = []
    case_number = db.query(models.Case).count()
    try:
        for index, row in enumerate(normalized, start=2):
            try:
                amount = float(_value(row, "amount") or "")
                if amount <= 0:
                    raise ValueError()
            except ValueError:
                raise HTTPException(400, f"Row {index}: amount must be a positive number.")

            customer_id = _value(row, "customer_id", "entity_id")
            record_id = _value(row, "claim_id", "transaction_id", "record_id", "id")
            record_id = record_id or f"UPLOAD-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{index}"
            existing = (db.query(models.Claim).filter_by(claim_id=record_id).first() if is_claim
                        else db.query(models.Transaction).filter_by(transaction_id=record_id).first())
            if existing:
                raise HTTPException(409, f"Row {index}: record ID {record_id} already exists.")
            raw_time = _value(row, "timestamp", "date", "event_time")
            try:
                timestamp = datetime.fromisoformat(raw_time.replace("Z", "+00:00")) if raw_time else datetime.now(timezone.utc)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            except ValueError:
                raise HTTPException(400, f"Row {index}: timestamp must be ISO date/time format.")
            label_raw = _value(row, "is_fraud", "fraud_label", "label")
            label = None if label_raw is None else label_raw.lower() in {"1", "true", "yes", "fraud"}
            known = {"amount", "customer_id", "entity_id", "customer_name", "transaction_id", "claim_id",
                "record_id", "id", "timestamp", "date", "event_time", "is_fraud", "fraud_label", "label",
                "account_id", "device_id", "merchant_id", "policy_id", "address_id", "broker_id", "email"}
            features = {k: v for k, v in row.items() if k and k not in known and v not in (None, "")}
            while True:
                case_number += 1
                case_id = f"FS-{case_number:05d}"
                if not db.query(models.Case).filter_by(case_id=case_id).first():
                    break
            if is_claim:
                db.add(models.Claim(claim_id=record_id, customer_id=customer_id,
                    policy_id=_value(row, "policy_id"), amount=amount, date=timestamp,
                    fraud_label=label, data_source="user_uploaded", raw_features=features))
            else:
                db.add(models.Transaction(transaction_id=record_id, customer_id=customer_id,
                    amount=amount, timestamp=timestamp, account_id=_value(row, "account_id"),
                    device_id=_value(row, "device_id"), merchant_id=_value(row, "merchant_id"),
                    is_fraud=label, data_source="user_uploaded", raw_features=features))
            case = models.Case(case_id=case_id, source="manual", domain=payload.domain,
                customer_id=customer_id, customer_name=_value(row, "customer_name"),
                claim_id=record_id if is_claim else None, transaction_id=None if is_claim else record_id,
                amount=amount, timestamp=timestamp, data_source="user_uploaded", risk_level="UNSCORED", status="new")
            db.add(case)
            created.append(case)
            graph_rows.append({"customer_id": customer_id, "name": _value(row, "customer_name") or "",
                "record_id": record_id, "domain": payload.domain, "amount": amount, "timestamp": timestamp.isoformat(),
                "account_id": _value(row, "account_id"), "device_id": _value(row, "device_id"),
                "address_id": _value(row, "address_id"), "broker_id": _value(row, "broker_id"), "email": _value(row, "email")})
        db.commit()
    except Exception:
        db.rollback()
        raise

    # Shared identifiers connect imported customers, giving case investigations
    # a persistent graph to inspect across multiple uploads.
    graph_warning = None
    graph = None
    try:
        graph = get_graph_store()
        graph.initialize_schema()
        for item in graph_rows:
            cid = item["customer_id"]
            if not cid:
                continue
            graph.add_customer(cid, item["name"], data_source="user_uploaded")
            record_type, rel = ("Claim", "FILED") if is_claim else ("Transaction", "MADE")
            graph.add_entity(record_type, item["record_id"], amount=item["amount"], timestamp=item["timestamp"])
            graph.link(rel, "Customer", cid, record_type, item["record_id"])
            for key, entity_type, relation in [("account_id", "Account", "OWNS_ACCOUNT"),
                ("device_id", "Device", "USES_DEVICE"), ("address_id", "Address", "HAS_ADDRESS"),
                ("broker_id", "Broker", "FILED_VIA"), ("email", "Email", "HAS_EMAIL")]:
                if item[key]:
                    graph.add_entity(entity_type, item[key])
                    graph.link(relation, "Customer", cid, entity_type, item[key])
    except Exception as exc:
        graph_warning = str(exc)
    finally:
        if graph:
            graph.close()

    return {"imported": len(created), "domain": payload.domain,
        "case_ids": [row.case_id for row in created], "graph_warning": graph_warning}
