# FraudShield — Phase Progress Log

This file is the running record of what's actually been built, tested, and
verified, phase by phase, per the project's own build rule: *never claim a
phase is done without running it.* Each entry below was written after that
phase's tests were actually executed in this environment.

| Phase | Name | Status | Verified by |
|---|---|---|---|
| 0 | Architecture & repo scaffold | ✅ Done | Directory structure created; `.env.example` written |
| 1 | Backend foundation (FastAPI/SQLite/SQLAlchemy/Pydantic) | ✅ Done | Manual case intake validation and linked transaction/claim records; live PostgreSQL 16.15 schema + ORM CRUD smoke passed |
| 2 | Dataset pipeline (IEEE-CIS + carclaims + synthetic scenarios) | ✅ Done | `pytest backend/tests/test_data_pipeline.py` — 16/16 passing; real row counts/fraud rates in `docs/DATASET.md` |
| 3 | Baseline fraud model | ✅ Done | `pytest backend/tests/test_baseline_models.py` — 10/10 passing; real metrics in `docs/EXPERIMENTS.md` |
| 4 | GNN fraud-ring detector | ✅ Sampled run | GraphSAGE trained on 20K nodes/split; test PR-AUC 0.2067, ROC-AUC 0.7683; checkpoint compatibility + forward pass tested; see `models/gnn/` and `docs/EXPERIMENTS.md` |
| 5 | Temporal anomaly model | ✅ Sampled run | LSTM trained on 30K rows/split; test PR-AUC 0.0486, ROC-AUC 0.6162; velocity detector rerun with LSTM feature; see `docs/temporal_lstm_metrics.json` |
| 6 | Document AI | ✅ Done | `test_document_ai.py` — 21/21 passing; **real tesseract OCR run end-to-end**; metrics in `docs/EXPERIMENTS.md` §7 |
| 7 | Graph store (Neo4j / kuzu) | ✅ Done | `test_graph.py` — 20/20 passing; real Cypher on kuzu 0.11.3; live Neo4j 5.26.32 constraints, CRUD, links, and fraud-ring queries passed |
| 8 | RAG knowledge base | ✅ Done | `test_rag.py` — 17/17 passing; **real ChromaDB 1.5.9**; accuracy@3 90.9%; metrics in `docs/EXPERIMENTS.md` §9 |
| 9 | Five agents + LangGraph | ✅ Done | `test_agents.py` — 28 passing; no-signal cases are explicitly unscored, human-reviewed, and not described as zero risk; **real LangGraph StateGraph, real tool calls, real LLM failure handled correctly**; 86.7% routing consistency on 30 synthetic cases; metrics in `docs/EXPERIMENTS.md` §11 |
| 10 | Explainability (SHAP / GNNExplainer) | ✅ | SHAP tests pass; GNNExplainer ran on held-out fraud node `3505452`; output in `docs/gnn_explanations.json` |
| 11 | Report generation (PDF) | ✅ Done | `test_report_generation.py` — 9/9 passing; **real ReportLab PDFs, verified via pdfplumber round-trip text extraction**; metrics in `docs/EXPERIMENTS.md` §15 |
| 12 | React dashboard | ✅ Done | Production build passes; global case-intake overlay preserves the active dashboard/graph view underneath; intake form and transaction/claim field switching were checked in browser; backend integration covers submit-to-investigation |
| 13 | WebSocket live updates | ✅ Done | `test_websocket.py` — 7/7 passing; **real LangGraph streaming, 2 real bugs found via live protocol/browser testing and fixed**; see `docs/EXPERIMENTS.md` §21 |
| 14 | Synthetic scenario engine | ✅ Done (built in Phase 2) | The spec's Phase 14 goal ('create controlled end-to-end cases') is exactly what Phase 2's `scripts/generate_synthetic_scenarios.py` already delivers — 8 scenarios, 3 CSVs, full provenance. No separate rebuild; see `docs/DATASET.md` §3. Interactive graph/document/agent execution against these scenarios is what Phases 7-13 each verified live. |
| 15 | Full integration | ✅ Done | `test_integration.py` — 13 passing; includes manual intake with no available signals routed to human review; seeded workflow verified live over real HTTP |
| 16 | Testing (unit/integration/e2e) | ✅ | 196 passed, 0 skipped in the current complete suite; current coverage not remeasured — see `docs/TESTING.md` |
| 17 | Docker | ✅ Done | `docker compose config`, image build, startup graph/RAG initialization, `/health`, dashboard HTTP, synthetic seed, and full case investigation verified live; see `docs/RUN_DOCKER.md` |
| 18 | Final documentation | ✅ Done | `docs/ARCHITECTURE.md` (with diagram), `docs/TESTING.md`, `docs/RUN_DOCKER.md` added this phase; all other docs updated incrementally throughout — see the full `docs/` directory |

Update this table at the end of every phase. Do not mark a phase ✅ unless
its tests were actually executed and passed in this session.

## Current cumulative test status

```
196 passed, 0 skipped, 0 failed
Coverage percentage not remeasured after the latest model changes.
  backend/tests/test_api.py .................... 12 passed
  backend/tests/test_data_pipeline.py .......... 16 passed
  backend/tests/test_baseline_models.py ........ 10 passed
  backend/tests/test_gnn.py .................... 14 passed
  backend/tests/test_temporal.py ............... 13 passed
  backend/tests/test_document_ai.py ............ 21 passed
  backend/tests/test_graph.py .................. 20 passed
  backend/tests/test_rag.py ..................... 17 passed
  backend/tests/test_agents.py .................. 28 passed
  backend/tests/test_explainability.py .......... 16 passed
  backend/tests/test_report_generation.py ....... 9 passed
  backend/tests/test_integration.py .............. 13 passed
  backend/tests/test_websocket.py ................ 7 passed
```

The complete suite was run after graph/RAG indexes were rebuilt; the GNN
and LSTM forward-pass checks and previously gated HTTP/WebSocket tests
all executed.

## Remaining external verification

Docker Compose has been built and smoke-tested on its SQLite/kuzu defaults.
PostgreSQL ORM CRUD and Neo4j graph queries passed direct live-service smoke
checks, but the complete API/Compose workflow has not been run with those
alternative backends. Gemini-backed responses require a valid
`GEMINI_API_KEY`; no key was provided. See `docs/EXPERIMENTS.md` for measured
neural/fusion results and the remaining dashboard/documentation gaps.
