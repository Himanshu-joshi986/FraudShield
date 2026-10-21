FROM python:3.12-slim

# tesseract-ocr: required for real OCR (Phase 6, backend/app/documents/ocr.py).
# Without it, TesseractBackend will fail at runtime with a clear error,
# not silently degrade -- see app/documents/ocr.py's degraded-mode handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    fonts-dejavu-core \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --default-timeout=180 --retries=5 -r backend/requirements.txt

# Optional heavy dependencies (GNN/LSTM/GNNExplainer, Gemini LLM client)
# are commented out in requirements.txt by design -- see that file's
# comments. Uncomment and rebuild if you've verified them per
# docs/RUN_GNN.md / docs/RUN_TEMPORAL.md.

COPY backend/ backend/
COPY scripts/ scripts/
COPY docs/ docs/
COPY knowledge_base/ knowledge_base/
COPY data/synthetic/ data/synthetic/
COPY data/documents/ data/documents/

# graph_db and chroma_db are built by scripts/ingest_graph.py and
# scripts/evaluate_rag.py respectively -- run once at container startup
# (see docker-compose.yml's backend command) rather than baked into the
# image, so the image doesn't need to bundle a stale index.

EXPOSE 8000

WORKDIR /app/backend
CMD ["sh", "-c", "python -m app.startup_checks && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
