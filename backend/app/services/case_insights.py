"""
Case insights -- the numbers behind a case, computed from what the
pipeline actually produced.

This module is the backend half of the dashboard's "Signal Scorecard",
"Policy trace", "Counterfactual" and "Case activity timeline" panels.
Everything it returns is derived from real persisted data:

  * the signal values and fusion weights come from Sentinel Agent's own
    recorded output (agent_executions.output_json),
  * the routing trace comes from Case Commander's recorded output,
  * the timeline comes from the Cases table plus the document manifest.

The scorecard decomposes the weighted average of available model and
heuristic signals. Evidence-policy adjustments are returned separately
so reviewers can distinguish that signal average from the final risk
assessment.

THE COUNTERFACTUALS ARE ARITHMETIC, NOT STORYTELLING
----------------------------------------------------
Fusion is a weighted mean of the available signals:

    risk_score = sum(v_i * w_i) / sum(w_i)

so the value a single signal would need in order to pull the score
under a given threshold is exactly solvable:

    x = (T * W - S + v_k * w_k) / w_k

The answer is reported with an honest caveat: if the case also trips a
policy OVERRIDE (ring, duplicate invoice, document mismatch, velocity),
then lowering the score alone would not change the routing -- and the
response says so rather than implying it would.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.database import models
from app.ml.fusion.risk_fusion import DEFAULT_WEIGHTS, RISK_LEVEL_THRESHOLDS
from app.agents.sentinel_agent import VELOCITY_WINDOW_SECONDS, VELOCITY_HIGH_COUNT
from app.agents.case_commander import HIGH_THRESHOLD, MEDIUM_THRESHOLD
from app.services import document_store
from app.services.velocity import as_utc, event_signature

TABULAR_BASELINE_REFERENCE = 0.05

SIGNAL_LABELS = {
    "tabular_ml_risk": "Tabular model",
    "graph_risk": "Graph connectivity",
    "temporal_risk": "Transaction velocity",
    "document_risk": "Document check",
}

# Why a signal can be missing. These are the honest, measured reasons --
# not placeholders.
SIGNAL_MISSING_NOTES = {
    "tabular_ml_risk": (
        "Not available: this case does not carry the IEEE-CIS feature vector the "
        "tabular model was trained on. The 0.05 baseline is a display-only reference, "
        "not a model prediction, and is excluded from the risk calculation."
    ),
    "graph_risk": "Not available: this investigation has no recorded graph score.",
    "temporal_risk": "Not available: this investigation has no recorded velocity score.",
    "document_risk": "Not available: no checked supporting document is linked to this case.",
}

OVERRIDE_PREFIXES = {
    "ring_override": "Coordinated-ring override",
    "duplicate_invoice_override": "Duplicate-invoice override",
    "document_mismatch_override": "Document amount-mismatch override",
    "velocity_override": "High-velocity override",
}


def _agent_output(executions, prefix: str) -> dict:
    for ex in executions:
        if (ex.agent_name or "").lower().startswith(prefix.lower()):
            out = ex.output_json
            return out if isinstance(out, dict) else {}
    return {}


def _iso(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return value


def _fmt_amount(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


# ---------------------------------------------------------------------------
# Signal scorecard
# ---------------------------------------------------------------------------

def _build_signals(sentinel: dict, evidence_rows=(), risk_score: float = 0.0) -> tuple[list[dict], dict]:
    # The per-signal values live in Sentinel's own execution record. If
    # that record predates the field (or the case was investigated by an
    # older build), fall back to the model_explanation exhibit, which
    # carries the same numbers in its source_reference.
    raw_signals = sentinel.get("signal_values")
    if not isinstance(raw_signals, dict):
        raw_signals = {}
    if not raw_signals:
        for item in evidence_rows:
            reference = item.source_reference if isinstance(item.source_reference, dict) else {}
            candidate = reference.get("signal_values")
            if isinstance(candidate, dict) and candidate:
                raw_signals = candidate
                break

    fusion = sentinel.get("fusion_detail") or {}
    weights = fusion.get("weights_used") or DEFAULT_WEIGHTS

    available = {k: v for k, v in raw_signals.items() if v is not None}
    total_weight = sum(weights.get(k, 0.0) for k in available) or 1.0

    rows = []
    for name in DEFAULT_WEIGHTS:
        value = raw_signals.get(name)
        weight = weights.get(name, DEFAULT_WEIGHTS[name])
        is_available = value is not None
        weighted = round(value * weight, 4) if is_available else None
        contribution = round(weighted / total_weight, 4) if is_available else None
        # Share of the base signal average, not of the weight pool: a
        # signal can carry 20% of weight and still dominate this case's score.
        # Reporting the weight share here would misread as "this barely
        # mattered" -- the opposite of the truth.
        share_of_score = (
            round(contribution / risk_score * 100, 1)
            if contribution is not None and risk_score > 0 else None
        )
        note = None if is_available else SIGNAL_MISSING_NOTES.get(name)
        if name == "temporal_risk" and is_available:
            event_count = sentinel.get("velocity_burst_count")
            window_seconds = sentinel.get("velocity_window_seconds", VELOCITY_WINDOW_SECONDS)
            if event_count is not None:
                note = (
                    f"{event_count} distinct events for this customer in the trailing "
                    f"{window_seconds // 60} minutes, including this case; high-velocity "
                    f"threshold is {VELOCITY_HIGH_COUNT}."
                )
        rows.append({
            "name": name,
            "label": SIGNAL_LABELS.get(name, name),
            "value": round(float(value), 4) if is_available else None,
            "available": is_available,
            "baseline_value": (
                TABULAR_BASELINE_REFERENCE
                if name == "tabular_ml_risk" and not is_available else None
            ),
            "weight": weight,
            "weighted": weighted,
            "contribution": contribution,
            "contribution_pct": share_of_score,
            "weight_share_pct": round(weight / total_weight * 100, 1) if is_available else None,
            "note": note,
        })
    return rows, fusion


# ---------------------------------------------------------------------------
# Policy trace
# ---------------------------------------------------------------------------

def _build_policy(commander: dict, decision_row) -> dict:
    conditions = commander.get("policy_conditions_triggered") or []
    overrides = []
    for condition in conditions:
        key = str(condition).split(":", 1)[0].strip()
        if key in OVERRIDE_PREFIXES:
            overrides.append(OVERRIDE_PREFIXES[key])

    return {
        "decision": (decision_row.decision if decision_row else commander.get("decision")),
        "reason": (decision_row.reason if decision_row else commander.get("reason")),
        "policy_version": (decision_row.policy_version if decision_row else commander.get("policy_version")),
        "policy_source": commander.get("policy_source"),
        "conditions_triggered": list(conditions),
        "overrides": overrides,
        # False when a policy override -- not the score -- drove the routing.
        "score_driven": not overrides,
    }


# ---------------------------------------------------------------------------
# Counterfactuals (exact arithmetic on the real fusion)
# ---------------------------------------------------------------------------

def _build_counterfactuals(signal_rows: list[dict], risk_score: float,
                            policy: dict) -> list[dict]:
    usable = [row for row in signal_rows if row["available"] and row["value"] is not None]
    if not usable:
        return []

    total_weight = sum(row["weight"] for row in usable)
    weighted_sum = sum(row["weighted"] for row in usable)

    targets = [(MEDIUM_THRESHOLD, "MEDIUM band"), (HIGH_THRESHOLD, "HIGH threshold")]
    out = []

    for target, target_label in targets:
        # A counterfactual is only meaningful for a threshold the case
        # currently EXCEEDS. Telling a reviewer what would happen if an
        # already-low score fell further is noise, not insight.
        if risk_score < target:
            continue

        for row in usable:
            required = (target * total_weight - weighted_sum + row["weighted"]) / row["weight"]
            required = round(required, 4)
            current = row["value"]

            if required < 0:
                explanation = (
                    f"{row['label']} could fall to 0 and the score would still be "
                    f"{max(0.0, round((weighted_sum - row['weighted']) / total_weight, 2))} "
                    f"-- this signal is not what put the case over {target_label}."
                )
                achievable = False
            elif required >= current:
                explanation = (
                    f"{row['label']} is already below the value that drives this case "
                    f"over {target_label}; changing it would not lower the score."
                )
                achievable = False
            else:
                explanation = (
                    f"{row['label']} would have to fall from {current:.2f} to "
                    f"{required:.2f} for the case to score below {target_label}."
                )
                achievable = True

            if policy.get("overrides"):
                explanation += (
                    " Note: an active policy override "
                    f"({', '.join(policy['overrides'])}) would still apply, so this "
                    "change alone would not change the routing outcome."
                )

            out.append({
                "signal": row["name"],
                "label": row["label"],
                "target_label": target_label,
                "target_threshold": target,
                "current_value": current,
                "required_value": required if required >= 0 else None,
                "achievable": achievable,
                "explanation": explanation,
            })

    # Only the most useful target per signal is worth showing first; keep
    # both but order them so the achievable MEDIUM ones lead.
    out.sort(key=lambda c: (not c["achievable"], c["target_threshold"]))
    return out


# ---------------------------------------------------------------------------
# Case activity timeline
# ---------------------------------------------------------------------------

def _build_events(db: Session, case_row: models.Case, manifest_cases: dict) -> tuple[list[dict], int]:
    customer_id = case_row.customer_id
    if customer_id:
        siblings = db.query(models.Case).filter(models.Case.customer_id == customer_id).all()
    else:
        siblings = [case_row]

    reference_time = case_row.timestamp
    events: list[dict] = []

    for case in siblings:
        kind = "claim" if case.domain == "insurance_claim" else "transaction"
        record_id = case.claim_id or case.transaction_id or case.case_id
        in_window = False
        if reference_time and case.timestamp:
            delta = (as_utc(reference_time) - as_utc(case.timestamp)).total_seconds()
            in_window = 0 <= delta <= VELOCITY_WINDOW_SECONDS

        events.append({
            "timestamp": _iso(case.timestamp),
            "kind": kind,
            "title": "Claim filed" if kind == "claim" else "Transaction posted",
            "detail": f"{case.case_id} · {_fmt_amount(case.amount)} · {case.customer_name or 'Unnamed entity'}",
            "case_id": case.case_id,
            "risk_level": case.risk_level,
            "is_current": case.case_id == case_row.case_id,
            "in_velocity_window": in_window,
            "flagged": case.risk_level == "HIGH",
        })

    # Supporting documents, filed alongside their case.
    for case in siblings:
        doc = manifest_cases.get(case.case_id)
        if not doc:
            continue
        invoice = doc.extracted.get("primary_invoice_number")
        events.append({
            "timestamp": _iso(case.timestamp),
            "kind": "document",
            "title": "Supporting document filed",
            "detail": f"Invoice {invoice or 'unreadable'} · {Path(doc.image_path).name}",
            "case_id": case.case_id,
            "risk_level": case.risk_level,
            "is_current": case.case_id == case_row.case_id,
            "in_velocity_window": False,
            "flagged": "duplicate_invoice_across_cases" in doc.flags,
        })

    events.sort(key=lambda e: (e["timestamp"] is None, e["timestamp"]))
    cluster_events = set()
    if customer_id:
        for case in siblings:
            if case.timestamp:
                delta = (as_utc(reference_time) - as_utc(case.timestamp)).total_seconds() if reference_time else None
                if delta is not None and 0 <= delta <= VELOCITY_WINDOW_SECONDS:
                    cluster_events.add(event_signature(case.customer_id, case.timestamp, case.amount, case.domain))
    cluster = len(cluster_events)
    return events, cluster


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_case_insights(db: Session, case_row: models.Case) -> dict:
    executions = sorted(case_row.agent_executions, key=lambda e: e.execution_id)
    sentinel = _agent_output(executions, "Sentinel")
    commander = _agent_output(executions, "Case Commander")
    decision_row = case_row.decisions[0] if case_row.decisions else None
    risk_score = float(case_row.risk_score or 0.0)

    fusion = sentinel.get("fusion_detail") or {}
    base_risk_score = float(fusion.get("base_risk_score", risk_score))
    signal_rows, fusion = _build_signals(sentinel, case_row.evidence, base_risk_score)
    policy = _build_policy(commander, decision_row)

    counterfactuals = _build_counterfactuals(signal_rows, base_risk_score, policy)

    doc_data = document_store.get_document_findings()
    events, cluster = _build_events(db, case_row, doc_data["findings"])

    agents = [{
        "agent_name": ex.agent_name,
        "status": ex.status,
        "duration_ms": ex.duration_ms,
        "started_at": _iso(ex.started_at),
        "completed_at": _iso(ex.completed_at),
        "summary": ex.output_summary,
    } for ex in executions]

    return {
        "case_id": case_row.case_id,
        "investigated": bool(executions),
        "risk_score": risk_score,
        "base_risk_score": base_risk_score,
        "risk_level": case_row.risk_level,
        "thresholds": {
            "MEDIUM": MEDIUM_THRESHOLD,
            "HIGH": HIGH_THRESHOLD,
            "risk_level_low": RISK_LEVEL_THRESHOLDS["LOW"],
            "velocity_window_seconds": VELOCITY_WINDOW_SECONDS,
            "velocity_high_count": VELOCITY_HIGH_COUNT,
        },
        "signals": signal_rows,
        "fusion": fusion,
        "policy": policy if executions else None,
        "counterfactuals": counterfactuals,
        # True when the score is already under the MEDIUM band -- in which
        # case any escalation came from a policy override, not the score,
        # and there is no score counterfactual worth showing.
        "score_below_medium": base_risk_score < MEDIUM_THRESHOLD,
        "events": events,
        "agents": agents,
        "velocity_window_seconds": VELOCITY_WINDOW_SECONDS,
        "velocity_cluster_size": cluster,
        "sibling_case_count": len({e["case_id"] for e in events if e.get("case_id")}),
    }


# ---------------------------------------------------------------------------
# Document dossier -- side-by-side invoice comparison
# ---------------------------------------------------------------------------

def _modal(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else ""


def _document_view(case_id: str, findings, manifest_by_case, is_current: bool) -> dict:
    finding = findings[case_id]
    entry = manifest_by_case.get(case_id, {})
    extracted = finding.extracted or {}
    printed = extracted.get("closest_document_amount")
    if printed is None:
        printed = extracted.get("max_amount")
    claim_amount = entry.get("claim_amount_on_file")
    dates = extracted.get("dates") or []
    return {
        "case_id": case_id,
        "customer_name": entry.get("customer_name"),
        "image_name": Path(finding.image_path).name,
        "invoice_number": extracted.get("primary_invoice_number"),
        "printed_amount": printed,
        "claim_amount_on_file": claim_amount,
        "document_date": dates[0] if dates else None,
        "ocr_backend": finding.ocr_backend,
        "ocr_confidence": finding.ocr_confidence,
        "document_risk": round(finding.document_risk, 4),
        "flags": list(finding.flags),
        "is_current_case": is_current,
        "degraded": finding.degraded,
        "degraded_reason": finding.degraded_reason,
    }


def build_document_dossier(case_id: str) -> dict:
    data = document_store.get_document_findings()
    findings = data["findings"]
    manifest_by_case = {entry["case_id"]: entry for entry in data["manifest"]}

    mine = findings.get(case_id)
    if mine is None:
        return {
            "case_id": case_id,
            "has_document": False,
            "group_invoice_number": None,
            "duplicate_across_cases": False,
            "documents": [],
            "rows": [],
            "summary": "No supporting document is attached to this case.",
        }

    invoice = (mine.extracted or {}).get("primary_invoice_number")
    group = [case_id]
    if invoice:
        group = sorted(data["duplicates"].get(invoice, [case_id]))
        if case_id not in group:
            group = sorted(group + [case_id])

    views = [_document_view(cid, findings, manifest_by_case, cid == case_id) for cid in group]

    # Build comparison rows, flagging the values that differ across the
    # group -- the modal value is the "expected" one, anything else is
    # the odd one out and gets highlighted.
    def row(label, getter, sub_getter=None, force_highlight=None):
        values = [getter(v) or "-" for v in views]
        modal = _modal(values)
        differs = len(set(values)) > 1
        cells = []
        for index, view in enumerate(views):
            highlight = force_highlight(view) if force_highlight else (differs and values[index] != modal)
            cells.append({
                "value": values[index],
                "sub": sub_getter(view) if sub_getter else None,
                "highlight": bool(highlight),
            })
        return {"label": label, "differs": differs, "cells": cells}

    rows = [
        row("Invoice number", lambda v: v["invoice_number"] or "-"),
        row("Customer name", lambda v: v["customer_name"] or "-"),
        row(
            "Amount printed",
            lambda v: _fmt_amount(v["printed_amount"]) if v["printed_amount"] is not None else "-",
            sub_getter=lambda v: f"record on file {_fmt_amount(v['claim_amount_on_file'])}"
            if v["claim_amount_on_file"] is not None else None,
            force_highlight=lambda v: "claim_amount_mismatch" in v["flags"],
        ),
        row("Document date", lambda v: v["document_date"] or "-"),
        row(
            "OCR confidence",
            lambda v: f"{v['ocr_confidence']:.2f}" if v["ocr_confidence"] is not None else "-",
        ),
        row("Flags", lambda v: ", ".join(v["flags"]) if v["flags"] else "none",
            force_highlight=lambda v: bool(v["flags"])),
    ]

    duplicate = len(group) > 1
    if duplicate:
        summary = (
            f"Invoice {invoice} was filed on {len(group)} separate cases "
            f"({', '.join(group)}). The same invoice appearing on more than one "
            "claim is a duplicate-filing pattern, not a coincidence."
        )
    elif "claim_amount_mismatch" in mine.flags:
        summary = (
            "The amount printed on this document does not match the recorded "
            "amount on file."
        )
    else:
        summary = "This document was read successfully and cross-checked against the claim record."

    return {
        "case_id": case_id,
        "has_document": True,
        "group_invoice_number": invoice,
        "duplicate_across_cases": duplicate,
        "documents": views,
        "rows": rows,
        "summary": summary,
    }
