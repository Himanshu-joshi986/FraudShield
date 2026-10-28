"""
Serves the supporting-document images referenced by the Phase-6 document
layer, so the dashboard can display the actual invoice next to the OCR
fields read from it.

SECURITY
--------
The endpoint is whitelisted against the document manifest's image
filenames rather than accepting a filesystem path. A request for any
name that is not a known document returns 404, so there is no path
traversal surface here at all -- no "..", no absolute paths, no arbitrary
file reads.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services import document_store

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("/{filename}")
def get_document_image(filename: str):
    """Return the document image for a known manifest entry."""
    if filename not in document_store.manifest_filenames():
        raise HTTPException(404, "No such document image")

    path = document_store.DOCUMENTS_DIR / Path(filename).name
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Document image is not present on disk")

    return FileResponse(path, media_type="image/png", filename=path.name)
