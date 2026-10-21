from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.core.config import settings
from app.database.session import init_db, engine
from app.api import cases, analytics, knowledge, data
from app.services.websocket_manager import manager as ws_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Warm the duplicate-invoice-detection cache at startup rather than
    # on the first investigate request: it runs OCR (tesseract) across
    # every document in the manifest, which took ~10.5s the first time it
    # ran -- found while testing the dashboard live, where it made the
    # first case investigation in a session feel broken/hung. Paying that
    # cost once at boot instead of on a user-facing request is the
    # correct fix, not just a faster one.
    from app.services.pipeline_service import _load_duplicate_invoice_case_ids
    _load_duplicate_invoice_case_ids()
    yield


app = FastAPI(
    title="FraudShield API",
    version="0.1.0-phase1",
    description="Multi-agent AI platform for explainable fraud detection and investigation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases.router)
app.include_router(analytics.router)
app.include_router(knowledge.router)
app.include_router(data.router)


@app.get("/health", tags=["health"])
def health():
    db_ok = True
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "app_env": settings.app_env, "db_connected": db_ok}


@app.get("/", tags=["health"])
def root():
    return {"service": "FraudShield API", "phase": "1 - backend foundation"}


@app.websocket("/ws/cases/{case_id}")
async def websocket_case_updates(websocket: WebSocket, case_id: str):
    """
    Section 40/41: real-time agent_started / agent_completed events for
    one case, broadcast from websocket_manager.run_case_streaming() as
    the actual LangGraph pipeline executes.
    """
    await ws_manager.connect(case_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive; client doesn't need to send anything meaningful
    except WebSocketDisconnect:
        ws_manager.disconnect(case_id, websocket)
