# How to Run the Temporal Models (Phase 5) — Step by Step

## What's already done vs. what you run

| Component | Status |
|---|---|
| Temporal feature engineering + IsolationForest (`velocity_detector.py`) | ✅ **Built and run on your real IEEE-CIS data** — results in `docs/EXPERIMENTS.md` §6 |
| LSTM sequence model (`sequence_model.py`) | ✅ Trained/evaluated; see `docs/temporal_lstm_metrics.json` |

---

## Part A — Re-run the velocity detector (no PyTorch needed)

This already ran at `--sample-size 30000`. Run it larger on your machine:

```bash
cd FraudShield
python3 backend/app/ml/temporal/velocity_detector.py --sample-size 100000

# full split (slower, needs more RAM):
python3 backend/app/ml/temporal/velocity_detector.py --sample-size 0
```

**Outputs:**
- `data/processed/temporal/{split}_temporal_features.parquet`
- `models/temporal/temporal_isolation_forest.joblib`
- `docs/temporal_metrics.json`

### Tuning the thresholds

All thresholds live at the top of `velocity_detector.py` and are recorded
into `docs/temporal_metrics.json` alongside whatever results they
produced, so a result is never separated from the settings that made it:

```python
VELOCITY_WINDOW_SECONDS = 3600   # trailing window for velocity counting
HIGH_VELOCITY_COUNT     = 5      # >=N txns in that window -> flag
RAPID_REFILE_SECONDS    = 300    # <N sec since entity's last txn -> flag
AMOUNT_SPIKE_ZSCORE     = 3.0    # amount >N SD above entity's own mean
```

If ~26% of transactions are getting flagged (as in the measured run) and
you want a tighter filter, raise `HIGH_VELOCITY_COUNT` to 8–10 and
`AMOUNT_SPIKE_ZSCORE` to 4.0, then re-run and compare
`fraud_rate_among_flagged` before/after. **Record both runs** — showing
you tuned a threshold and measured the effect is a genuinely good thing
to put in your report.

---

## Part B — Train the LSTM sequence model

### Step 1 — Install PyTorch

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
# or with a GPU (check nvidia-smi for your CUDA version):
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 2 — Prerequisite

Part A must have run first — the LSTM consumes the engineered temporal
features it produces.

### Step 3 — Train and evaluate the LSTM

The reproducible trainer is `scripts/train_temporal.py`. It consumes the
temporal parquet artifacts from Part A, fits feature scaling on train
only, selects by validation PR-AUC, and evaluates the selected weights
once on validation and test.

```bash
python scripts/train_temporal.py --epochs 20 --patience 5 --batch-size 256
```

Outputs: `models/temporal/fraud_lstm.pt` and
`docs/temporal_lstm_metrics.json`. The checked-in bounded run used 30,000
rows per split, CPU, sequence length 10, hidden size 64, and two LSTM
layers. Its test PR-AUC is 0.0486 and ROC-AUC is 0.6162. This is an LSTM,
not a Temporal Fusion Transformer; see the measured result in
`docs/EXPERIMENTS.md`.

### Step 4 — Tuning knobs

| Knob | Effect |
|---|---|
| `seq_len=10` → `20` | Model sees longer history; better for slow-building patterns, more memory |
| `hidden_size=64` → `128` | More capacity; risks overfitting on 3.5% positives |
| `num_layers=2` → `1` | Faster, less overfitting, less expressive |
| `batch_size=256` → `1024` | Faster on GPU; may need a higher `lr` |

---

## How to report these results honestly

**Do not call the LSTM a Temporal Fusion Transformer.** The module says
so in its own docstring and there's a test enforcing that the disclaimer
stays (`test_module_states_it_is_not_a_TFT`). A TFT is a substantially
different architecture aimed at multi-horizon forecasting. If you want a
real TFT, use `pytorch-forecasting`'s implementation — don't relabel this.

**Expect modest standalone numbers, and say so.** The measured
IsolationForest version scored ROC-AUC 0.55–0.62 — barely above random as
a standalone classifier. That is not a failure; it's the honest
characterisation of an unsupervised timing-anomaly detector. Its real
contribution is as an **independent signal in risk fusion** (Phase 17):
flagged transactions carry ~1.6× the fraud rate of unflagged ones, and
that lift comes from information the tabular model doesn't have. Report
the lift, not an inflated AUC.

**If the LSTM also scores modestly, report that too.** On IEEE-CIS, entity
sequences are often short (many entities appear only once or twice), which
structurally limits what a sequence model can learn. Checking the
distribution of `entity_txn_seq` before blaming the model is worth doing —
and worth writing up.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `build_sequences` uses lots of RAM | It materialises `(n, seq_len, n_features)` floats. At 100K rows × 10 × 6 that's ~24MB — fine. At `seq_len=50` it's 5× that. Lower `seq_len` or the sample size. |
| Loss not decreasing | Lower `lr` to 1e-4; confirm `class_weighted_loss` is being used |
| Recall stuck at 0 | Class weighting missing or the fraud count in your sample is ~0 — check `y.sum()` |
| `inf` values crash training | `build_sequences` already sanitises inf (there's a test for it); if you bypass it, replace `seconds_since_prev`'s inf for first-ever transactions |
| Most entities have 1 transaction | Real property of IEEE-CIS — consider filtering to entities with ≥3 transactions and documenting that you did |
