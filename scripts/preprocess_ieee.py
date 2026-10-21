"""
Phase 2 -- IEEE-CIS Fraud Detection dataset pipeline.

Real data in, real numbers out: this script loads the actual Kaggle
train_transaction.csv + train_identity.csv files supplied in
data/raw/ieee_cis/, engineers features, and produces a **time-aware**
train/validation/test split (fraud is a temporal phenomenon -- a random
shuffle-split would leak future information into training, which is a
classic and serious mistake on this dataset).

Memory strategy: this environment has ~3.9GB RAM. train_transaction.csv
is 394 columns x 590,540 rows. Loaded naively as float64 that's already
~1.9GB just for the numeric matrix, before categorical/string overhead.
So every column is read with an explicit, minimal dtype from the start
(float32 for numerics, category for the small set of categorical
columns, int8 for the label) rather than loaded then downcast.

Note: Kaggle's test_transaction.csv/test_identity.csv have NO isFraud
label (it's the competition's private leaderboard holdout) -- they
cannot be used for evaluation. We only use train_transaction/train_identity,
splitting them ourselves into train/val/test.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "ieee_cis"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]
IDENTITY_CATEGORICAL_COLS = [
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]


def _build_dtype_map(columns, categorical_cols):
    dtype_map = {}
    for c in columns:
        if c in ("TransactionID",):
            dtype_map[c] = "int32"
        elif c == "isFraud":
            dtype_map[c] = "int8"
        elif c in categorical_cols:
            dtype_map[c] = "category"
        elif c == "TransactionDT":
            dtype_map[c] = "int32"
        else:
            dtype_map[c] = "float32"
    return dtype_map


def load_raw():
    t0 = time.time()
    txn_cols = pd.read_csv(RAW_DIR / "train_transaction.csv", nrows=0).columns.tolist()
    txn_dtypes = _build_dtype_map(txn_cols, CATEGORICAL_COLS)
    txn = pd.read_csv(RAW_DIR / "train_transaction.csv", dtype=txn_dtypes)
    print(f"Loaded train_transaction: {txn.shape} in {time.time()-t0:.1f}s, "
          f"{txn.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    t0 = time.time()
    id_cols = pd.read_csv(RAW_DIR / "train_identity.csv", nrows=0).columns.tolist()
    id_dtypes = _build_dtype_map(id_cols, IDENTITY_CATEGORICAL_COLS)
    identity = pd.read_csv(RAW_DIR / "train_identity.csv", dtype=id_dtypes)
    print(f"Loaded train_identity: {identity.shape} in {time.time()-t0:.1f}s, "
          f"{identity.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return txn, identity


def merge_and_engineer(txn: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    df = txn.merge(identity, on="TransactionID", how="left", indicator=True)
    has_identity = (df["_merge"] == "both").astype("int8").rename("has_identity")
    df = df.drop(columns=["_merge"])

    # --- temporal features from TransactionDT (seconds since a reference point) ---
    txn_day = (df["TransactionDT"] // (24 * 3600)).astype("int32").rename("txn_day")
    txn_hour = ((df["TransactionDT"] // 3600) % 24).astype("int8").rename("txn_hour")
    txn_dayofweek = (txn_day % 7).astype("int8").rename("txn_dayofweek")

    # --- amount features ---
    amt_log = np.log1p(df["TransactionAmt"].astype("float64")).astype("float32").rename("TransactionAmt_log")

    # --- frequency-encoding for high-cardinality identifiers (fraud-ring style signal:
    #     unusually high reuse of the same card/address is itself informative) ---
    freq_cols = []
    for col in ["card1", "addr1"]:
        freq = df[col].map(df[col].value_counts()).astype("float32").rename(f"{col}_freq")
        freq_cols.append(freq)

    # build all new columns at once via concat to avoid the fragmented-frame
    # performance penalty of many individual `df[col] = ...` assignments
    new_cols = pd.concat([has_identity, txn_day, txn_hour, txn_dayofweek, amt_log, *freq_cols], axis=1)
    df = pd.concat([df, new_cols], axis=1)

    return df


def time_aware_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """
    Sorts by TransactionDT and splits chronologically -- train on the past,
    validate/test on the future. This mirrors how the model would actually
    be deployed (scoring transactions that happen after the training
    period) and avoids the data leakage a random shuffle-split would cause.
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def main():
    txn, identity = load_raw()
    df = merge_and_engineer(txn, identity)
    del txn, identity  # free memory before split/save

    train, val, test = time_aware_split(df)

    stats = {
        "total_rows": int(len(df)),
        "n_features": int(df.shape[1]),
        "identity_coverage_pct": round(float(df["has_identity"].mean() * 100), 2),
        "splits": {},
    }
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df.to_parquet(OUT_DIR / f"ieee_cis_{name}.parquet", index=False)
        stats["splits"][name] = {
            "rows": int(len(split_df)),
            "fraud_count": int(split_df["isFraud"].sum()),
            "fraud_rate_pct": round(float(split_df["isFraud"].mean() * 100), 3),
            "dt_range": [int(split_df["TransactionDT"].min()), int(split_df["TransactionDT"].max())],
        }
        print(f"{name}: {stats['splits'][name]}")

    with open(OUT_DIR / "ieee_cis_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nSaved to", OUT_DIR)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
