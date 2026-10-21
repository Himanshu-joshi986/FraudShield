# Architecture

This diagram reflects what's built. GraphSAGE and the LSTM are trained on
bounded IEEE-CIS samples, but their checkpoints are used by offline
evaluation scripts only; manually entered cases do not consume IEEE-CIS
feature vectors. SQLite, PostgreSQL ORM CRUD, kuzu, Neo4j adapter queries,
and ChromaDB have been smoke-tested. A full API deployment against
PostgreSQL plus Neo4j has not been verified.

```mermaid
flowchart TB
    subgraph Frontend["React Dashboard (Phase 12)"]
        UI[Command Center / Manual Case Intake / Case Detail / Analytics / Knowledge Base]
    end

    subgraph API["FastAPI Backend (Phase 1, wired Phase 15)"]
        REST["/api/cases, /api/analytics, /api/knowledge"]
        WS["/ws/cases/{id} -- live agent events (Phase 13)"]
    end

    subgraph Agents["5-Agent LangGraph Pipeline (Phase 9)"]
        Sentinel["Sentinel Agent (no LLM)"]
        Investigation["Investigation Agent (real tools, LLM w/ fallback)"]
        Evidence["Evidence Analyst (no LLM)"]
        Report["Report Architect (LLM narrative w/ fallback)"]
        Commander["Case Commander (no LLM, policy rules)"]
        Sentinel -->|risk gate| Investigation
        Investigation --> Evidence --> Report --> Commander
    end

    subgraph Detection["Detection Layers"]
        Baseline["Tabular ML (Phase 3): trained on real IEEE-CIS+carclaims"]
        GNN["Graph construction + trained GraphSAGE (20K nodes/split)"]
        Temporal["Velocity detector + trained LSTM (30K rows/split)"]
        DocAI["Document AI (Phase 6): real tesseract OCR"]
        Fusion["Risk Fusion (Phase 9/17)"]
    end

    subgraph Stores["Data Stores"]
        SQL[("SQLite verified / PostgreSQL ORM smoke-tested")]
        Graph[("kuzu verified / Neo4j adapter smoke-tested")]
        Vector[("ChromaDB RAG (Phase 8)")]
    end

    subgraph Explain["Explainability (Phase 10)"]
        SHAP["SHAP: run on real trained models"]
        GNNExp["GNNExplainer: run on a held-out fraud node"]
        ExplainCase["Explain This Case: verified live"]
    end

    subgraph Output["Report Generation (Phase 11)"]
        PDF["ReportLab PDF: verified via text round-trip"]
    end

    UI <-->|HTTP| REST
    UI <-.->|WebSocket, real-time| WS
    REST --> Agents
    WS -.->|agent_started/completed| Agents

    Sentinel --> Fusion
    Fusion --> Baseline
    Fusion --> GNN
    Fusion --> Temporal
    Fusion --> DocAI

    Investigation <--> Graph
    Investigation <--> Vector
    Evidence --> Explain
    Report --> PDF

    Agents --> SQL
    REST --> SQL
```

## Request flow: investigating one case

1. **User opens a case** in the dashboard or posts it to `POST /api/cases`. Optional transaction/claim IDs are stored as linked source records. A submitted case is then investigated from its detail view (or through `POST /api/cases/{id}/investigate`).
2. **FastAPI** loads the case, opens the graph store (kuzu) and knowledge base (ChromaDB) singletons.
3. **LangGraph's `.stream()`** runs the 5 agents in order, yielding after each one completes.
4. Each yielded step is **immediately broadcast** over the case's WebSocket subscribers (`agent_started` / `agent_completed`) — this is genuine per-node streaming, not simulated progress (see the real concurrency bug found and fixed in Phase 13, `docs/EXPERIMENTS.md` §21).
5. **Sentinel** queries the graph store for connections, checks for a matching document (OCR'd if present), computes a velocity heuristic from case timestamps, and fuses whichever signals are actually available into a risk score (`risk_fusion.py`).
6. If at least one risk signal is available, risk is negligible, and there's zero graph connectivity, the pipeline **short-circuits** straight to an auto-close decision — skipping the (rate-limited) LLM calls entirely. If there are no usable signals, the case remains unscored and routes to human review; missing evidence is not treated as low risk.
7. Otherwise, **Investigation** queries the graph again for the full connection list, queries the RAG knowledge base for relevant regulatory passages, and attempts an LLM synthesis (Gemini) that **falls back to a deterministic four-part template** if the LLM is unreachable — which it always is in the sandboxed dev environment (verified: 403 Forbidden), so this fallback path has been exercised for real, not just written.
8. **Evidence Analyst** compiles every finding into a structured, source-referenced evidence list.
9. **Report Architect** builds the structural report entirely from that evidence list (never inventing content), plus an optional LLM narrative paragraph.
10. **Case Commander** applies the routing policy (4 override rules, each recording its own trigger condition) and produces the final decision.
11. **All of the above is persisted**: `AgentExecution`, `Evidence`, `Report`, `Decision` rows, plus a generated PDF file.
12. The final `investigation_complete` WebSocket event fires, and the dashboard re-fetches the case detail.

## Why this shape, not a simpler one

- **Sentinel runs first and can short-circuit** specifically so the expensive, rate-limited LLM calls (free-tier Gemini: as low as 10 RPM / 500 RPD) are never spent on cases with no signal at all — a real constraint that shaped the architecture, not an afterthought.
- **Every "no LLM" agent is deliberately deterministic** (Sentinel, Evidence Analyst, Case Commander) — the spec explicitly rejects "5 chained LLM calls" as an anti-pattern, and this project takes that literally: only 2 of 5 agents ever call an LLM at all, and both have working fallbacks.
- **Risk fusion normalizes by available signal weight**, not a fixed denominator — a case that only has graph + document signals (no tabular ML score, because it's a synthetic scenario fixture rather than an IEEE-CIS-shaped transaction) still produces a correctly-scaled 0–1 score instead of one silently capped low.
- **The graph and SQL stores have configuration surfaces for switching backends** (kuzu↔Neo4j and SQLite↔Postgres); only kuzu and SQLite have been exercised. ChromaDB runs in embedded/persistent mode.
