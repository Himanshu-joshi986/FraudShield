# Dataset Documentation

## 1. IEEE-CIS Fraud Detection (primary financial fraud benchmark)

- **Source**: [Kaggle — IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- **License**: Kaggle competition data, subject to competition rules; not redistributable outside this project's evaluation use.
- **Files used**: `train_transaction.csv`, `train_identity.csv` (user-supplied). `test_transaction.csv`/`test_identity.csv` were also supplied but **cannot be used for evaluation** — Kaggle withholds the `isFraud` label for the test set (it's the private leaderboard holdout).

### Measured statistics (computed directly from the files, not assumed)

| | Rows | Columns |
|---|---|---|
| `train_transaction.csv` | 590,540 | 394 |
| `train_identity.csv` | 144,233 | 41 |
| `test_transaction.csv` (unlabeled, unused for training/eval) | 506,691 | 393 |

- **Fraud rate (train_transaction)**: 3.499% (20,663 fraud / 590,540 total) — a real, heavily imbalanced rate, consistent with published descriptions of this dataset.
- **Identity coverage**: 24.42% of transactions have a matching identity record (144,233 / 590,540). The majority of transactions have no device/email/browser data — this is a real characteristic of the dataset, not a processing error, and the pipeline includes a `has_identity` flag so downstream models/agents can account for it.

### Preprocessing (`scripts/preprocess_ieee.py`)

- Every column read with an explicit minimal dtype (float32 for numerics, category for the ~16 known categorical columns, int8 for the label) from the start — loading this file naively as float64 would use ~1.9GB for the numeric matrix alone before overhead, which doesn't comfortably fit this environment's 3.9GB RAM.
- Transaction + identity merged on `TransactionID` (left join, since most transactions lack identity data).
- Engineered features: `txn_hour`, `txn_dayofweek` (from `TransactionDT`), `TransactionAmt_log`, `card1_freq`/`addr1_freq` (frequency-encoding — repeated use of the same card/address is itself an informative fraud-ring-style signal).

### Split strategy: time-aware, not random

Fraud is a temporal phenomenon (fraud patterns and countermeasures shift over time), so a random shuffle-split would leak future information into training. Rows are sorted by `TransactionDT` and split chronologically 70/15/15:

| Split | Rows | Fraud count | Fraud rate |
|---|---|---|---|
| Train | 413,378 | 14,464 | 3.499% |
| Validation | 88,581 | 3,083 | 3.481% |
| Test | 88,581 | 3,116 | 3.518% |

Fraud rate is stable across all three splits (no distribution drift), and it was verified that `max(train.TransactionDT) <= min(val.TransactionDT) <= min(test.TransactionDT)` — no leakage (see `backend/tests/test_data_pipeline.py::TestIEEECISPipeline::test_no_temporal_leakage_between_splits`).

---

## 2. Vehicle Insurance Fraud Claims (carclaims.csv)

- **Source**: user-supplied `carclaims.csv` — a well-known public vehicle-insurance-fraud benchmark (commonly distributed as the "Vehicle Claim Fraud Detection" dataset).
- **Records**: 15,420 claims, 33 columns.
- **Target**: `FraudFound` (Yes/No), mapped to `fraud_label` (1/0) during preprocessing.
- **Fraud rate**: 5.986% (923 / 15,420) — measured directly, not assumed.
- **Features**: demographic/categorical (Make, AccidentArea, Sex, MaritalStatus, Fault, PolicyType, VehicleCategory, PastNumberOfClaims, PoliceReportFiled, WitnessPresent, AgentType, etc.) plus a handful of numeric fields (Age, WeekOfMonth, Deductible, DriverRating, Year). No raw transaction amounts or precise timestamps are present.

### Preprocessing (`scripts/preprocess_carclaims.py`)

- `fraud_label` derived from `FraudFound`.
- Categorical columns cast to `category` dtype (23 columns).
- Synthetic `claim_id` minted (`CC-000000` ...) since `PolicyNumber` is reused across multiple claim-years in this dataset and is not a reliable unique row identifier.

### Split strategy: stratified random (not time-aware)

Unlike IEEE-CIS, this dataset provides no reliable per-row chronological ordering (only Month/Year/WeekOfMonth buckets, and rows aren't guaranteed to be in temporal order), so a time-aware split isn't meaningful here. A stratified random split (70/15/15, `random_state=42`) is used instead, preserving the fraud rate across splits:

| Split | Rows | Fraud count | Fraud rate |
|---|---|---|---|
| Train | 10,793 | 646 | 5.985% |
| Validation | 2,314 | 139 | 6.007% |
| Test | 2,313 | 138 | 5.966% |

### Documented limitation

`carclaims.csv` provides no exact claim date, only Month/Year/WeekOfMonth buckets — precise per-day temporal features (as engineered for IEEE-CIS) are **not possible** with this dataset. This is stated explicitly rather than approximated silently.

---

## 3. Synthetic Scenario Library

**This is not training data.** Per Section 8 of the spec, synthetic data here is used *only* for controlled investigation scenarios the public benchmarks above cannot provide in usable form: multi-entity fraud-ring relationships, document mismatch/duplication, and scenarios with a labeled "expected outcome" for testing the agent pipeline's behavior end-to-end. No model is trained on this data, and it must never be described as representing real-world fraud.

Generated by `scripts/generate_synthetic_scenarios.py`, seeded (`random.seed(7)`) for reproducibility. Three CSVs, all under `data/synthetic/`:

| File | Rows | Purpose |
|---|---|---|
| `synthetic_cases.csv` | 26 | One row per case (transaction/claim), across all 8 scenarios |
| `synthetic_relationships.csv` | 42 | Entity-relationship graph edges (owns_account, uses_device, has_email, has_address, filed_via_broker) |
| `synthetic_documents.csv` | 8 | Claim/invoice document pairs, including deliberate amount mismatches and duplicate invoice numbers |

Every row carries `data_source=synthetic` and a `scenario_id` (`FS-SC-001` through `FS-SC-008`), matching Section 8's required provenance metadata.

### The 8 scenarios (Section 9)

| Scenario | Name | What it tests | Expected risk |
|---|---|---|---|
| FS-SC-001 | Legitimate case | No false positives on clean data | LOW |
| FS-SC-002 | Transaction velocity anomaly | 9-transaction burst in ~50 min, same account | MEDIUM |
| FS-SC-003 | Fraud ring | 6 entities sharing account/device/email/broker | HIGH |
| FS-SC-004 | Document mismatch | Claim $5,000 vs. invoice $7,500 | MEDIUM |
| FS-SC-005 | Duplicate invoice | Same invoice number reused across 3 claims | HIGH |
| FS-SC-006 | Shared identity | 4 "unrelated" claimants sharing one address | MEDIUM |
| FS-SC-007 | Mixed evidence | One weak signal only — tests against over-flagging | MEDIUM |
| FS-SC-008 | High-risk connected case | 4 independent signals agree (temporal + graph + document + duplicate) | HIGH |

Verified programmatically (`backend/tests/test_data_pipeline.py::TestSyntheticScenarios`) — e.g. Scenario 3's graph actually has an entity shared by all 6 ring members, Scenario 4's document actually has a mismatched amount, Scenario 5's invoice number actually repeats. These aren't just labeled rows; the underlying structure matches the label.

---

## 4. Limitations (documented per Section 63)

- IEEE-CIS and carclaims are both several years old; fraud patterns and countermeasures evolve, so current-day performance would need revalidation on fresher data.
- IEEE-CIS's `test_transaction.csv` cannot be used for evaluation (no label available) — all real metrics in this project come from the labeled `train_transaction.csv`, split ourselves into train/val/test.
- carclaims lacks precise per-claim timestamps, limiting temporal feature engineering for that domain.
- The synthetic scenario library is deliberately small (26 cases) and hand-constructed to exercise specific agent behaviors — it is a **test fixture**, not a statistically representative sample of anything, and is never used to report detection-accuracy metrics.
- Both public datasets reflect their originating population/geography/era and may not generalize to other markets or time periods without revalidation.
