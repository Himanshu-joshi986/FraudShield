# How to Run the GNN (Phase 4) — Step by Step

## Current status

The graph *construction* layer was built and run on your real IEEE-CIS
The checked-in GraphSAGE checkpoint was trained on CPU using 20,000
nodes per split. Its held-out test PR-AUC is 0.2067 and ROC-AUC is
0.7683. The checkpoint and metrics are in `models/gnn/`; the run and
comparison with XGBoost are recorded in `docs/EXPERIMENTS.md` §5.

---

## Step 1 — Install PyTorch and PyTorch Geometric

**CPU-only (works on any laptop, slower):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

**With an NVIDIA GPU (much faster — recommended if you have one):**

```bash
# check your CUDA version first with: nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
```

Verify:

```bash
python3 -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
python3 -c "import torch_geometric; print('PyG', torch_geometric.__version__)"
```

If `CUDA: False` on a machine with a GPU, you installed the CPU build —
reinstall with the matching `cu###` index URL.

---

## Step 2 — Build the graph (no PyTorch needed)

This step already ran successfully in the build environment at
`--sample-size 20000`. Re-run it at a larger size on your machine:

```bash
cd FraudShield

# Development run (fast, ~1 min):
python3 backend/app/ml/gnn/graph_builder.py --sample-size 20000

# Fuller run (recommended for your final results, slower):
python3 backend/app/ml/gnn/graph_builder.py --sample-size 100000

# Full split (only if you have 16GB+ RAM and time — see warning below):
python3 backend/app/ml/gnn/graph_builder.py --sample-size 0
```

⚠️ **Warning about `--sample-size 0`**: building transaction-to-transaction
adjacency is quadratic in the number of transactions sharing each entity.
On the full 413K-row training split this can take a very long time and use
a lot of RAM. Start at 100,000 and only increase if it completes
comfortably. **Whatever sample size you use, write it down** — a GNN
trained on a 20K sample is not comparable to one trained on 400K, and
reporting them as if they were would be exactly the kind of fake
comparison the spec forbids.

**Outputs** (in `data/processed/graph/`):
- `{split}_node_features.parquet` — 26 features per transaction
- `{split}_node_labels.parquet` — isFraud label
- `{split}_edges.parquet` — transaction↔transaction edges
- `{split}_node_meta.parquet` — TransactionID, proxy_uid, timestamp
- `{split}_graph_stats.json` — node/edge counts, avg degree
- `{split}_fraud_rings.json` — connected components ≥3 transactions

---

## Step 3 — Train the GNN

```bash
python3 scripts/train_gnn.py --epochs 100 --hidden 64
```

Useful flags:
- `--epochs 200` — train longer (early stopping will cut it off if val PR-AUC stops improving)
- `--hidden 128` — bigger model (needs more memory)
- `--lr 0.001` — default learning rate; lower it if loss is unstable
- `--patience 25` — wait longer before early-stopping

**Expected runtime**: minutes on a GPU; tens of minutes to a couple hours
on CPU at 100K nodes, depending on your machine.

**Outputs**:
- `models/gnn/gnn_fraudsage.pt` — trained weights
- `models/gnn/gnn_metrics.json` — full metrics
- `docs/gnn_metrics.json` — identical copy for the experiment record

---

## Step 4 — Explain a held-out node and compare signal fusion

Once the checkpoint exists:

```bash
python scripts/evaluate_gnn_explainer.py --epochs 30
python scripts/evaluate_fusion.py
```

The first command saves a mapped node/edge explanation to
`docs/gnn_explanations.json`. The second requires the trained LSTM
(`python scripts/train_temporal.py`) and baseline XGBoost artifact; it
compares all scores on shared test transaction IDs and writes
`docs/fusion_metrics.json`. It does not tune weights on the test set.

## Step 5 — Record your results honestly

After each run, check both metric files and update the experiment table.
Training overwrites the current checkpoint and metrics with the new run;
keep the selected sample size and settings in the record.

Then compare it against the Phase-3 baseline, which was measured on the
same dataset:

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| XGBoost (Phase 3 baseline, full 88K test set) | 0.886 | 0.471 |
| Your GNN | ? | ? |

**Important — read this before you write a conclusion:**

1. **If the GNN scores lower than XGBoost, say so.** That is a completely
   legitimate, publishable finding, and it is a much stronger project than
   a fabricated win. Graph methods do not automatically beat gradient
   boosting on tabular fraud data — on IEEE-CIS specifically, well-tuned
   GBDTs are very strong, and a 2-layer GraphSAGE on 26 features may not
   beat them. What the GNN offers that XGBoost cannot is **ring-level
   structure**: identifying *groups* of connected fraudulent transactions
   (see the 17 all-fraud rings found in Step 2), which is the actual
   investigative value, not the raw AUC.
2. **Compare like with like.** If you trained the GNN on a 20K sample,
   don't compare it to the XGBoost number measured on the full 88K test
   set. Either note the difference explicitly, or re-evaluate XGBoost on
   the same subset.
3. **Report the full metric set**, not just the best-looking one — PR-AUC
   matters far more than ROC-AUC at 3.5% fraud prevalence.

---

## Step 5 — Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Graph too large for GPU | Lower `--sample-size` in Step 2, or `--hidden 32` |
| Training loss goes to NaN | Learning rate too high | `--lr 0.001` |
| Recall stays at 0.0 | Class imbalance dominating | Already handled by `class_weighted_loss`; if it persists, check the fraud count in `{split}_graph_stats.json` is non-zero |
| `--sample-size 0` never finishes | Quadratic adjacency blowup | Use 100000; or raise `--min-shared-entities 3` to cut edges |
| Very few edges built | Entity-degree filter too aggressive | Raise `--max-entity-degree 2000` |
| `ModuleNotFoundError: app.ml.gnn` | Wrong working directory | Run from the `FraudShield/` repo root |

---

## What comes after

Once the GNN is trained, its per-transaction fraud probability feeds the
**risk fusion layer** (Phase 17) alongside the tabular, temporal, and
document signals — no single model decides a case. The fraud rings from
Step 2 feed the **Investigation Agent** (Phase 9) and the **Fraud Ring
Explorer** dashboard page (Phase 12).

`GNNExplainer` (Phase 10) will run on this same trained model to produce
the per-case edge/node importance explanations the Evidence Analyst needs.
