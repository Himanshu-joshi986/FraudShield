"""
Phase 2 -- Vehicle insurance claims fraud dataset pipeline.

Source: data/raw/carclaims/carclaims.csv (uploaded by the user).
This is a well-known public vehicle-insurance-fraud benchmark: 15,420
claims, 33 columns, target `FraudFound` (Yes/No), 5.99% fraud rate
(computed directly from the file below -- not assumed).

Unlike IEEE-CIS, this dataset has no reliable per-row timestamp (only
Month/Year/WeekOfMonth buckets, and rows are not guaranteed to be in
chronological order), so a time-aware split isn't meaningful here.
We use a stratified random split instead, preserving the fraud rate
across train/val/test -- documented explicitly as a different splitting
strategy from IEEE-CIS's, per Section 46's requirement to document the
split strategy per dataset.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "carclaims" / "carclaims.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = [
    "Month", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed", "MonthClaimed",
    "Sex", "MaritalStatus", "Fault", "PolicyType", "VehicleCategory", "VehiclePrice",
    "Days:Policy-Accident", "Days:Policy-Claim", "PastNumberOfClaims", "AgeOfVehicle",
    "AgeOfPolicyHolder", "PoliceReportFiled", "WitnessPresent", "AgentType",
    "NumberOfSuppliments", "AddressChange-Claim", "NumberOfCars", "BasePolicy",
]


def main():
    df = pd.read_csv(RAW_PATH)

    df["fraud_label"] = (df["FraudFound"].str.strip().str.lower() == "yes").astype("int8")
    df = df.drop(columns=["FraudFound"])

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    # unique claim_id for downstream FraudShield case linkage (PolicyNumber
    # is not unique per row in this dataset re-used across multiple claim
    # years, so we mint a synthetic sequential claim_id instead)
    df.insert(0, "claim_id", [f"CC-{i:06d}" for i in range(len(df))])
    df["data_source"] = "carclaims"

    train_val, test = train_test_split(df, test_size=0.15, stratify=df["fraud_label"], random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, stratify=train_val["fraud_label"], random_state=42)
    # 0.1765 * 0.85 ≈ 0.15 -> final split is ~70/15/15

    stats = {"total_rows": int(len(df)), "fraud_rate_pct": round(float(df["fraud_label"].mean() * 100), 3), "splits": {}}
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df.to_parquet(OUT_DIR / f"carclaims_{name}.parquet", index=False)
        stats["splits"][name] = {
            "rows": int(len(split_df)),
            "fraud_count": int(split_df["fraud_label"].sum()),
            "fraud_rate_pct": round(float(split_df["fraud_label"].mean() * 100), 3),
        }
        print(f"{name}: {stats['splits'][name]}")

    with open(OUT_DIR / "carclaims_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
