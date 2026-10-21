"""
Phase 5a -- Temporal / velocity anomaly detection (NO PyTorch REQUIRED).

Per Section 15 of the spec, the temporal layer must output:

    {"temporal_risk": 0.84, "anomalies": ["high_transaction_velocity", ...]}

This module builds per-entity transaction sequences from IEEE-CIS and
computes the temporal features that define velocity fraud: how fast is
this entity transacting, how much has the amount jumped versus its own
history, is it transacting at an unusual hour for itself.

IMPORTANT HONESTY NOTE (Section 15 explicitly requires this):
This module implements **engineered temporal features + an
IsolationForest anomaly detector**. It is NOT a Temporal Fusion
Transformer, and nothing in this project should claim it is. The
sequence-model (LSTM) version is provided separately in
ml/temporal/sequence_model.py for you to train -- see docs/RUN_TEMPORAL.md.
Both are real; this one is simply the one that runs without a GPU.

Entity identity: IEEE-CIS has no customer_id, so sequences are grouped by
the same proxy uid used in the graph layer (card1 + addr1 + email domain).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSED = REPO_ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "temporal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds are configurable and documented rather than magic numbers
# buried in code (Build Rule 19: "make thresholds configurable").
VELOCITY_WINDOW_SECONDS = 3600          # 1 hour
HIGH_VELOCITY_COUNT = 5                 # >=5 txns in the window
RAPID_REFILE_SECONDS = 300              # <5 min since same entity's last txn
AMOUNT_SPIKE_ZSCORE = 3.0               # amount >3 SD above entity's own mean
UNUSUAL_HOUR_START, UNUSUAL_HOUR_END = 1, 5   # 01:00-05:00 local-ish


def build_proxy_uid(df: pd.DataFrame) -> pd.Series:
    """Must match ml/gnn/graph_builder.build_proxy_uid exactly so the graph
    and temporal layers agree on what an 'entity' is."""
    return (
        df["card1"].astype("float").fillna(-1).astype("int64").astype(str)
        + "_" + df["addr1"].astype("float").fillna(-1).astype("int64").astype(str)
        + "_" + df["P_emaildomain"].astype("object").fillna("unknown").astype(str)
    )


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-transaction temporal features computed *within each entity's own
    history*, sorted by time. All rolling statistics are strictly
    backward-looking (shift(1) before expanding) so a transaction never
    sees its own value in its own baseline -- that would be target leakage.
    """
    df = df.copy()
    df["proxy_uid"] = build_proxy_uid(df)
    df = df.sort_values(["proxy_uid", "TransactionDT"]).reset_index(drop=True)

    g = df.groupby("proxy_uid", sort=False)

    # seconds since this entity's previous transaction
    df["seconds_since_prev"] = g["TransactionDT"].diff()
    df["seconds_since_prev"] = df["seconds_since_prev"].fillna(np.inf)

    # entity's transaction index (1st, 2nd, 3rd... transaction)
    df["entity_txn_seq"] = g.cumcount() + 1

    # backward-looking mean/std of this entity's amounts (excludes current row)
    prev_amounts = g["TransactionAmt"].shift(1)
    df["entity_amt_mean_prior"] = prev_amounts.groupby(df["proxy_uid"]).expanding().mean().reset_index(level=0, drop=True)
    df["entity_amt_std_prior"] = prev_amounts.groupby(df["proxy_uid"]).expanding().std().reset_index(level=0, drop=True)

    # z-score of current amount vs this entity's own prior history
    std = df["entity_amt_std_prior"].replace(0, np.nan)
    df["amt_zscore_vs_own_history"] = (
        (df["TransactionAmt"] - df["entity_amt_mean_prior"]) / std
    ).fillna(0.0)

    # transactions by this entity within the trailing velocity window
    def count_in_window(group):
        times = group["TransactionDT"].to_numpy()
        counts = np.searchsorted(times, times, side="right") - np.searchsorted(
            times, times - VELOCITY_WINDOW_SECONDS, side="left"
        )
        return pd.Series(counts, index=group.index)

    df["txns_in_trailing_hour"] = g.apply(count_in_window, include_groups=False).reset_index(level=0, drop=True)

    df["is_unusual_hour"] = df["txn_hour"].between(UNUSUAL_HOUR_START, UNUSUAL_HOUR_END).astype("int8")

    return df


TEMPORAL_FEATURE_COLS = [
    "seconds_since_prev", "entity_txn_seq", "amt_zscore_vs_own_history",
    "txns_in_trailing_hour", "is_unusual_hour", "TransactionAmt_log",
]


def flag_anomalies(row) -> list[str]:
    """
    Rule-based anomaly flags, matching the spec's required output shape.
    These are deliberately interpretable -- the Evidence Analyst needs to
    cite a named anomaly ("high_transaction_velocity"), not an opaque score.
    """
    flags = []
    if row["txns_in_trailing_hour"] >= HIGH_VELOCITY_COUNT:
        flags.append("high_transaction_velocity")
    if row["seconds_since_prev"] < RAPID_REFILE_SECONDS:
        flags.append("rapid_successive_transactions")
    if abs(row["amt_zscore_vs_own_history"]) > AMOUNT_SPIKE_ZSCORE:
        flags.append("sudden_amount_change")
    if row["is_unusual_hour"] == 1:
        flags.append("unusual_hour_activity")
    return flags


class TemporalAnomalyDetector:
    """
    IsolationForest over the engineered temporal features, producing the
    `temporal_risk` score (0-1) the risk-fusion layer consumes.

    Unsupervised by design: it learns what 'normal timing behaviour' looks
    like across the population rather than fitting the fraud label, so it
    can flag novel velocity patterns the supervised models haven't seen.
    Its fraud-detection performance is evaluated against the label
    afterwards (see fit_and_evaluate) rather than optimized for it.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=100, contamination=contamination,
            random_state=random_state, n_jobs=1,
        )
        self._score_min = None
        self._score_max = None

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        X = df[TEMPORAL_FEATURE_COLS].copy()
        # inf appears for an entity's first-ever transaction (no previous txn)
        X["seconds_since_prev"] = X["seconds_since_prev"].replace(np.inf, 1e7)
        return X.fillna(0.0).to_numpy(dtype="float32")

    def fit(self, df: pd.DataFrame):
        X = self._prepare(df)
        self.model.fit(X)
        raw = self.model.decision_function(X)
        self._score_min, self._score_max = float(raw.min()), float(raw.max())
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Returns temporal_risk in [0,1]; higher = more anomalous."""
        X = self._prepare(df)
        raw = self.model.decision_function(X)
        rng = (self._score_max - self._score_min) or 1.0
        normalized = (self._score_max - raw) / rng
        return np.clip(normalized, 0.0, 1.0)


def fit_and_evaluate(sample_size: int | None = 100000):
    """
    Fits on train, evaluates temporal_risk against the fraud label on
    val/test. Honest framing: an unsupervised anomaly detector is NOT
    expected to match a supervised classifier's AUC -- its value is
    catching timing patterns as an *independent signal* for risk fusion.
    We measure it so we know how much it actually contributes.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    results = {}
    detector = None
    stats = {"sample_size": sample_size, "thresholds": {
        "velocity_window_seconds": VELOCITY_WINDOW_SECONDS,
        "high_velocity_count": HIGH_VELOCITY_COUNT,
        "rapid_refile_seconds": RAPID_REFILE_SECONDS,
        "amount_spike_zscore": AMOUNT_SPIKE_ZSCORE,
    }, "splits": {}}

    for split in ["train", "val", "test"]:
        df = pd.read_parquet(PROCESSED / f"ieee_cis_{split}.parquet")
        if sample_size and sample_size < len(df):
            df = df.sort_values("TransactionDT").head(sample_size)

        feat = engineer_temporal_features(df)

        if split == "train":
            detector = TemporalAnomalyDetector().fit(feat)

        feat["temporal_risk"] = detector.score(feat)
        feat["anomaly_flags"] = feat.apply(flag_anomalies, axis=1)

        y = feat["isFraud"].to_numpy()
        risk = feat["temporal_risk"].to_numpy()

        flagged = feat["anomaly_flags"].str.len() > 0
        split_stats = {
            "rows": int(len(feat)),
            "roc_auc": round(float(roc_auc_score(y, risk)), 4),
            "pr_auc": round(float(average_precision_score(y, risk)), 4),
            "pct_with_any_flag": round(float(flagged.mean() * 100), 2),
            "fraud_rate_among_flagged": round(float(feat.loc[flagged, "isFraud"].mean()), 4) if flagged.any() else 0.0,
            "fraud_rate_among_unflagged": round(float(feat.loc[~flagged, "isFraud"].mean()), 4) if (~flagged).any() else 0.0,
        }
        stats["splits"][split] = split_stats
        print(f"{split}: {json.dumps(split_stats)}")

        feat[["TransactionID", "proxy_uid", "TransactionDT", "isFraud",
              "temporal_risk", "txn_hour"] + TEMPORAL_FEATURE_COLS].to_parquet(
            OUT_DIR / f"{split}_temporal_features.parquet", index=False
        )

    import joblib
    models_dir = REPO_ROOT / "models" / "temporal"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, models_dir / "temporal_isolation_forest.joblib")

    with open(REPO_ROOT / "docs" / "temporal_metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=100000,
                        help="rows per split (0 = full split)")
    args = parser.parse_args()
    fit_and_evaluate(sample_size=None if args.sample_size == 0 else args.sample_size)
