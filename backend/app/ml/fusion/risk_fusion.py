"""
Phase 9 / 17 -- Risk fusion (Section 17).

Combines whichever signals are actually available for a given case into
one risk_score, rather than letting a single model decide.

IMPORTANT ARCHITECTURAL NOTE, stated plainly rather than glossed over:
The Phase-3 tabular ML model (XGBoost) was trained on IEEE-CIS's specific
69-feature vector (TransactionAmt_log, C1-C14, D-columns, etc.). The
Phase-2 synthetic scenario library's case records do NOT carry that
feature vector -- they are lightweight investigation-scenario fixtures
(customer_id, amount, timestamp, scenario_id), not IEEE-CIS transaction
rows. So for the synthetic demo scenarios that the 5-agent pipeline runs
against in this phase, the tabular ML signal is legitimately not
available, and fusion combines only the signals that ARE available for
that case (graph, document) -- it does not fabricate a tabular score. For
a real production transaction carrying the full IEEE-CIS-shaped feature
vector, the tabular signal would also be available and included.

This is why `fuse()` normalizes by the sum of *available* signal
weights, not a fixed denominator -- a case with only 2 of 4 possible
signals still produces a well-scaled 0-1 score rather than one silently
capped at 0.5.
"""
from __future__ import annotations

# Default weights -- configurable, documented (Build Rule 19), NOT claimed
# to be optimal (Section 17: "Do not arbitrarily claim that a particular
# weighting is optimal"). These are a reasonable starting point pending
# the calibration Phase 17's full evaluation would do with more case data.
DEFAULT_WEIGHTS = {
    "tabular_ml_risk": 0.35,
    "graph_risk": 0.30,
    "temporal_risk": 0.15,
    "document_risk": 0.20,
}

RISK_LEVEL_THRESHOLDS = {"LOW": 0.35, "MEDIUM": 0.65}  # HIGH is anything >= MEDIUM's threshold


def fuse(signals: dict[str, float | None], weights: dict[str, float] | None = None) -> dict:
    """
    signals: {signal_name: value in [0,1] or None if not available for this case}
    Returns {"risk_score": float, "risk_level": str, "available_signals": [...],
             "missing_signals": [...], "weights_used": {...}}
    """
    weights = weights or DEFAULT_WEIGHTS

    available = {k: v for k, v in signals.items() if v is not None and k in weights}
    missing = [k for k in weights if k not in available]

    if not available:
        return {
            "risk_score": 0.0, "risk_level": "UNSCORED",
            "available_signals": [], "missing_signals": list(weights.keys()),
            "weights_used": {}, "degraded": True,
            "degraded_reason": "no signals available for this case",
        }

    total_weight = sum(weights[k] for k in available)
    weighted_sum = sum(available[k] * weights[k] for k in available)
    risk_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    risk_score = round(min(1.0, max(0.0, risk_score)), 4)

    if risk_score >= RISK_LEVEL_THRESHOLDS["MEDIUM"]:
        risk_level = "HIGH"
    elif risk_score >= RISK_LEVEL_THRESHOLDS["LOW"]:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "available_signals": sorted(available.keys()),
        "missing_signals": missing,
        "weights_used": {k: weights[k] for k in available},
        "degraded": False,
    }
