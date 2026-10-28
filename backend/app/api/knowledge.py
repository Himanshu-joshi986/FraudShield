"""
Knowledge base endpoints (Section 40): GET /api/knowledge,
POST /api/knowledge/query.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.rag.knowledge_base import get_knowledge_base
from app.database import models
from app.database.session import get_db

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class DocumentLink(BaseModel):
    case_id: str | None = None


@router.get("")
def list_knowledge_base():
    kb = get_knowledge_base()
    return {"documents": kb.list_documents(), "stats": kb.stats()}


@router.post("/query")
def query_knowledge_base(question: str, top_k: int = 3, case_id: str | None = None):
    kb = get_knowledge_base()
    return {"question": question, "results": kb.query(question, top_k=max(1, min(top_k, 10)), case_id=case_id)}


@router.post("/documents", status_code=201)
async def upload_knowledge_document(
    file: UploadFile = File(...),
    title: str = Form(""),
    jurisdiction: str = Form("Unspecified"),
    case_id: str = Form(""),
    db: Session = Depends(get_db),
):
    if case_id and not db.query(models.Case).filter_by(case_id=case_id).first():
        raise HTTPException(404, f"Case {case_id} does not exist.")
    content = await file.read(10 * 1024 * 1024 + 1)
    try:
        document = get_knowledge_base().add_upload(
            file.filename or "reference.txt", content, title, jurisdiction, case_id or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, "The document could not be indexed. No upload was kept.") from exc
    return {"document": document, "message": "Document saved and indexed for search."}


@router.put("/documents/{document_id}/link")
def link_knowledge_document(document_id: str, payload: DocumentLink, db: Session = Depends(get_db)):
    if payload.case_id and not db.query(models.Case).filter_by(case_id=payload.case_id).first():
        raise HTTPException(404, f"Case {payload.case_id} does not exist.")
    try:
        document = get_knowledge_base().link_upload(document_id, payload.case_id)
    except KeyError as exc:
        raise HTTPException(404, "User-provided document not found.") from exc
    except Exception as exc:
        raise HTTPException(500, "The case link could not be indexed. The previous link was restored.") from exc
    return {"document": document, "message": "Case link saved."}


@router.delete("/documents/{document_id}")
def delete_knowledge_document(document_id: str):
    try:
        document = get_knowledge_base().remove_upload(document_id)
    except KeyError as exc:
        raise HTTPException(404, "User-provided document not found.") from exc
    except Exception as exc:
        raise HTTPException(500, "The document could not be removed. Its saved copy was retained.") from exc
    return {"document": document, "message": "Document, extracted text, and indexed passages removed."}
