# Testing Strategy (Phase 16)

This project's testing was built incrementally, phase by phase, rather
than as a single pass at the end — every phase from 1 through 13 has its
own test file, written and run against real artifacts as that phase was
built (see `docs/PROGRESS.md` for the phase-by-phase count). This document
consolidates that into one picture, per Section 58's requirement for
unit, integration, and end-to-end coverage.

## Summary

```
196 passed, 0 skipped, 0 failed
```

The 71% coverage figure recorded before the model work has not been
re-measured against this updated suite; do not treat it as the current
coverage percentage. The current complete backend run is
`python -m pytest backend/tests -q` (196 passed).

**Read the 71% precisely, not just as a headline number** — it measures
code executed *within the pytest process*. Three categories of code in
this repo are tested differently, and coverage.py can only see one of them:

### 1. In-process tested (covered directly by pytest — the 71%)

Backend API, database models, all 5 agents, risk fusion, RAG, graph
store, explainability, report generation, WebSocket streaming. These are
called directly by test functions, so `coverage.py` sees every line
executed. This is real, standard test coverage.

### 2. Artifact-tested but NOT in-process (shows as 0% coverage, but IS verified)

Three modules show 0% in the coverage report:
- `app/ml/gnn/graph_builder.py`
- `app/ml/temporal/velocity_detector.py`
- `app/documents/generator.py`

These are run as **standalone scripts**, not imported and called by the
pytest suite. Their *output artifacts* (parquet files, generated document
images) ARE tested — `test_gnn.py`, `test_temporal.py`, and
`test_document_ai.py` load those real artifacts and assert real
properties about them (no leakage, correct fraud-ring structure, correct
OCR extraction, etc.). But because the module's own code never executes
inside the pytest process, `coverage.py` cannot credit it. **0% coverage
here does not mean untested — it means tested a different way**, and
conflating the two would either understate what's verified or, worse,
overstate it by claiming these lines "passed" when coverage never touched
them. Both would be dishonest; this is the precise version.

### 3. Neural model execution and explainability

GraphSAGE and the LSTM have been trained on bounded real IEEE-CIS samples.
The test suite now performs the PyTorch/PyG forward-pass checks and checks
that the saved GraphSAGE checkpoint matches the repository architecture.
GNNExplainer was run on one held-out fraudulent transaction; its mapped
explanation is saved in `docs/gnn_explanations.json`. These checks verify
execution and artifact compatibility, not general explanation fidelity
or generalization beyond the stated sample sizes.

## Breakdown by test file

| File | Tests | What it verifies |
|---|---|---|
| `test_api.py` | 12 | Phase 1 — case CRUD, health check, manual intake validation and linked source records |
| `test_data_pipeline.py` | 16 | Phase 2 — real IEEE-CIS/carclaims splits, no leakage, synthetic scenario structure |
| `test_baseline_models.py` | 10 | Phase 3 — real trained model metrics, beats-random-chance sanity checks |
| `test_gnn.py` | 14 | Phase 4 — real graph construction, forward pass, checkpoint compatibility |
| `test_temporal.py` | 13 | Phase 5 — velocity features, LSTM sequence building and forward pass |
| `test_document_ai.py` | 21 | Phase 6 — real OCR (tesseract), field extraction, duplicate detection |
| `test_graph.py` | 20 | Phase 7 — real Cypher queries on kuzu, fraud-ring detection |
| `test_rag.py` | 17 | Phase 8 — real ChromaDB retrieval, provenance, a documented TF-IDF limitation |
| `test_agents.py` | 28 | Phase 9 — risk fusion, no-signal handling/evidence, LLM degradation handling, all 5 agents + full pipeline evaluation |
| `test_explainability.py` | 16 | Phase 10 — real SHAP, GNNExplainer artifact, "Explain This Case" |
| `test_report_generation.py` | 9 | Phase 11 — real PDF generation, verified via text round-trip extraction |
| `test_integration.py` | 13 | Phase 15 — full stack through real HTTP, including manual intake with no signals routed to human review |
| `test_websocket.py` | 7 | Phase 13 — real LangGraph streaming; regression guard for a real concurrency bug |

## What "unit / integration / end-to-end" means concretely here

- **Unit**: individual agent functions, risk fusion math, amount-parsing edge cases, LLM client budget logic — fast, no external dependencies, hand-constructed state.
- **Integration**: `test_integration.py` exercises the full FastAPI app (real routes, real database, real graph store, real RAG index) via `TestClient`, without a live server process.
- **End-to-end**: beyond the automated suite, the dashboard was verified with a live backend + live frontend + a real browser (Playwright) and a real WebSocket client — see `docs/EXPERIMENTS.md` §19 and §21, including two real bugs (a shared-mutable-state concurrency bug and a missing WebSocket proxy config) found specifically *because* live end-to-end testing was done, not skipped in favor of unit/integration tests alone.

## Live alternative-database smoke checks

These checks were run separately from the complete test suite against disposable
services on the Docker network:

- PostgreSQL 16.15: SQLAlchemy connected using `postgresql+psycopg`,
  created the application schema, and passed a `Case` ORM insert/read/delete.
- Neo4j 5.26.32: the adapter initialized uniqueness constraints, created
  customers and a shared device, and returned the expected two-hop
  connection and shared-device cluster before cleaning up its test nodes.

These validate the adapters' basic live operations, not a full application
deployment or API investigation with PostgreSQL and Neo4j configured.

## A specific pattern worth naming directly

Several tests in this suite exist specifically to lock in a bug that was
found and fixed, with the docstring explaining what regression it guards
against — not generic "does it run" assertions. Examples:
- `test_case_commander_address_only_does_not_trigger_ring_override` (Phase 9)
- `test_streaming_events_are_distinct_per_agent` (Phase 13)
- `test_known_paraphrase_failure_is_recorded` (Phase 8 — locks in a known *limitation*, not a bug, so it can't be silently "fixed" by curve-fitting)
- `test_known_open_gap_is_documented` (Phase 9 — same pattern, for an intentionally-unresolved calibration question)

This is a deliberate pattern throughout the project: tests that encode
what was actually learned while building it, not just tests that happen
to pass.
