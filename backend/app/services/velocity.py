"""Shared temporal helpers for the case-level velocity signal and timeline."""
from __future__ import annotations

from datetime import datetime, timezone


def as_utc(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if not isinstance(parsed, datetime):
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def event_signature(customer_id, timestamp, amount, domain="") -> tuple:
    """Collapse duplicate ingestions of the same customer event conservatively."""
    stamp = as_utc(timestamp)
    try:
        amount_key = round(float(amount or 0.0), 2)
    except (TypeError, ValueError):
        amount_key = 0.0
    return (str(customer_id or ""), stamp.isoformat() if stamp else "", amount_key, str(domain or ""))
