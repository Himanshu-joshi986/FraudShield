"""
Knowledge base endpoints (Section 40): GET /api/knowledge,
POST /api/knowledge/query.
"""
from fastapi import APIRouter

from app.rag.knowledge_base import get_knowledge_base

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get("")
def list_knowledge_base():
    kb = get_knowledge_base()
    return {"documents": kb.list_documents(), "stats": kb.stats()}


@router.post("/query")
def query_knowledge_base(question: str, top_k: int = 3):
    kb = get_knowledge_base()
    return {"question": question, "results": kb.query(question, top_k=top_k)}
