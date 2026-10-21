# How to Run FraudShield with Docker — Step by Step

## Verification status

| Component | Status |
|---|---|
| Compose configuration | ✅ Parsed with `docker compose config --quiet` |
| Backend and frontend images | ✅ Built successfully |
| Backend initialization | ✅ Graph and RAG indices built on first start |
| Backend `/health` | ✅ HTTP 200, SQLite connected |
| Frontend `/` | ✅ HTTP 200 through nginx |
| Synthetic seed and investigation | ✅ Seeded 30 cases; one investigation completed through all five agents, persisting evidence and report |
| PostgreSQL / Neo4j adapters | ✅ Live schema/CRUD and graph-query smoke checks passed; full Compose/API workflow still uses and is verified on SQLite/kuzu |

The first live container start exposed an image-layout error: startup
looked for `/scripts/ingest_graph.py` outside the copied repository tree.
The backend image now copies the required project files under `/app` and
runs from `/app/backend`. Compose waits for backend health before starting
the frontend, avoiding a race while startup checks build the indices.

---

## Step 1 — Build and smoke-test

```bash
docker compose config --quiet
docker compose build
docker compose up -d
```

Watch the backend container's logs for the startup check:

```
[startup] Graph store not found, building it now (scripts/ingest_graph.py)...
[startup] RAG index not found, building it now (scripts/evaluate_rag.py)...
[startup] Checks complete.
```

Verify the services and run a small API smoke test:

```bash
curl http://localhost:8000/health
curl http://localhost:3000/
curl -X POST http://localhost:8000/api/cases/seed-synthetic
curl -X POST http://localhost:8000/api/cases/FS-SYN-00030/investigate
curl http://localhost:8000/api/cases/FS-SYN-00030/evidence
```

The verified run returned `{"status":"ok","db_connected":true}` from
`/health`, HTTP 200 from nginx, and seeded 30 cases. The sample
investigation completed through five agents, produced 11 evidence items,
and persisted its report. Without `GEMINI_API_KEY`, the Investigation
Agent records its normal LLM-degraded warning and uses the deterministic
fallback.

## Step 2 — Persistent data

The `fraudshield_data` named volume holds the SQLite database, the kuzu
graph store, and the ChromaDB index. It survives `docker compose down`.
Use `docker compose down -v` only when you intentionally want to erase the
database and force a clean index initialization on next `up`.

## Step 3 — Enable the LLM (optional)

```bash
export GEMINI_API_KEY=your-key-here
docker compose up
```

Without a key, Investigation and Report Architect use their deterministic
fallback automatically (same behavior verified throughout this project's
development — see `docs/EXPERIMENTS.md` §11).

## Step 4 — Alternative database adapters

The commented block at the bottom of `docker-compose.yml` shows the
services and environment variables. The application has configuration
and adapter code for `FRAUDSHIELD_DB_URL`, `GRAPH_BACKEND`, and `NEO4J_*`
(see `backend/app/core/config.py` and `backend/app/graph/store.py`).
Direct adapter smoke checks passed against PostgreSQL 16.15 and Neo4j
5.26.32: PostgreSQL schema creation and ORM CRUD, plus Neo4j constraints,
customer/device CRUD, relationship creation, connected-customer lookup,
and shared-entity clustering.

These checks do not constitute a full API/Compose run on PostgreSQL and
Neo4j. After switching, re-run Step 1's health, seed, and investigation
checks against the selected services; the Compose deployment documented
above remains verified on SQLite/kuzu.

## Step 5 — GNN / LSTM model runtime

The Docker image does not include PyTorch by default (see
`backend/requirements.txt`'s commented-out heavy dependencies). If you
want to train and serve the GNN or LSTM models inside a container, you'll
need a separate dependency and runtime configuration. The trained
checkpoints are currently evaluated offline; inference is not wired into
the synthetic-case pipeline.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Backend container exits immediately | Startup check failed (check logs for which script) | Run `scripts/ingest_graph.py` / `scripts/evaluate_rag.py` manually inside the container to see the real error |
| Frontend loads but API calls fail | nginx `/api` proxy misconfigured, or backend not ready yet | Check `docker/nginx.conf`; confirm `curl http://backend:8000/health` works from inside the frontend container |
| WebSocket live updates don't work | Missing `Upgrade`/`Connection` headers on the `/ws` proxy location | This is exactly the Phase 13 bug (see `docs/EXPERIMENTS.md` §21) — `docker/nginx.conf` already includes the fix; verify it wasn't edited out |
| OCR fails inside the container | `tesseract-ocr` not installed in the image | Confirm `docker/backend.Dockerfile`'s `apt-get install` line ran during build (check build logs, not just the final image) |
| Slow first investigation | Startup check hasn't finished pre-warming (Phase 12's fix) | Wait for `[startup] Checks complete.` in the logs before the first request |
