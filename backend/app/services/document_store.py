"""
Shared, cached access to the Phase-6 document layer (real OCR findings).

Why this module exists
----------------------
The document evidence in FraudShield comes from real OCR (tesseract) over
the invoice PNGs in data/documents/. Running OCR is EXPENSIVE: reading all
9 manifest documents took ~10.5 seconds the first time it ran, which was
found while testing the dashboard live (that cost used to be paid on every
investigate call -- it is now paid once at startup instead).

Two places need those findings:

  1. the investigation pipeline (cross-case duplicate-invoice detection)
  2. the dashboard's document dossier (side-by-side invoice comparison)

If each of them ran its own OCR pass, the dashboard would pay the cost
again on every request. So both read from this one cache.

CACHE INVALIDATION
------------------
The cache is keyed on the modification times of the manifest AND of every
document image it references. Re-running OCR is therefore correct the
moment a document changes on disk, and skipped entirely when nothing has
changed -- the best of both worlds, rather than the permanently-stale
module-level cache this replaces.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.documents.validator import validate_document, detect_duplicate_documents

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS_DIR = REPO_ROOT / "data" / "documents"
MANIFEST_PATH = DOCUMENTS_DIR / "manifest.json"

# OCR is the slowest thing in the document layer, and the documents are
# independent of each other -- so they are read concurrently rather than
# one after another. The engine is shared and thread-safe.
_OCR_WORKERS = 4

_lock = threading.Lock()
_cache: dict | None = None


def load_manifest() -> list[dict]:
    """Raw manifest entries, or an empty list when no manifest exists."""
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []


def manifest_filenames() -> set[str]:
    """
    The set of image basenames the API is allowed to serve.

    Serving is whitelisted against this set rather than resolving an
    arbitrary path, so the image endpoint can never be used to read a
    file outside data/documents/ (no path traversal surface at all).
    """
    names = set()
    for entry in load_manifest():
        path = entry.get("image_path")
        if path:
            names.add(Path(path).name)
    return names


def _cache_key(manifest: list[dict]) -> tuple:
    stamps: list[float] = []
    try:
        stamps.append(MANIFEST_PATH.stat().st_mtime)
    except OSError:
        stamps.append(0.0)
    for entry in manifest:
        try:
            stamps.append((REPO_ROOT / entry["image_path"]).stat().st_mtime)
        except (OSError, KeyError):
            stamps.append(0.0)
    return tuple(stamps)


def get_document_findings() -> dict:
    """
    Returns {
        "findings":  {case_id: DocumentFindings},
        "duplicates": {invoice_number: [case_id, ...]},   # invoices on 2+ cases
        "manifest":  [manifest entries],
    }

    OCR runs only when the manifest or one of its documents has changed.
    """
    global _cache

    manifest = load_manifest()
    key = _cache_key(manifest)

    with _lock:
        if _cache is not None and _cache["key"] == key:
            return _cache["data"]

        def read(entry):
            return validate_document(
                image_path=REPO_ROOT / entry["image_path"],
                claim_amount=entry["claim_amount_on_file"],
                claimant_name=entry.get("customer_name", ""),
                case_id=entry["case_id"],
            )

        if manifest:
            with ThreadPoolExecutor(max_workers=_OCR_WORKERS) as pool:
                # map() preserves input order, so the findings list stays
                # in manifest order regardless of which read finishes first.
                findings_list = list(pool.map(read, manifest))
        else:
            findings_list = []

        duplicates = detect_duplicate_documents(findings_list)

        data = {
            "findings": {f.case_id: f for f in findings_list},
            "duplicates": duplicates,
            "manifest": manifest,
        }
        _cache = {"key": key, "data": data}
        return data


def duplicate_invoice_case_ids() -> set[str]:
    """Case ids that share an invoice number with at least one other case."""
    findings = get_document_findings()
    return {
        case_id
        for case_id, f in findings["findings"].items()
        if "duplicate_invoice_across_cases" in f.flags
    }


def findings_for_case(case_id: str):
    """DocumentFindings for one case, or None when the case has no document."""
    return get_document_findings()["findings"].get(case_id)


def cases_sharing_invoice(case_id: str) -> list[str]:
    """Other case ids that filed the same invoice number as this case."""
    findings = get_document_findings()["findings"]
    mine = findings.get(case_id)
    if mine is None:
        return []
    invoice = mine.extracted.get("primary_invoice_number")
    if not invoice:
        return []
    return [cid for cid in get_document_findings()["duplicates"].get(invoice, []) if cid != case_id]
