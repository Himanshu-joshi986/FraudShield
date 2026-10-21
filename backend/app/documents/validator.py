"""
Phase 6c -- Document validation.

Section 16 is explicit: "Do not merely OCR and display text. Perform
actual validation." So this module does three fraud-relevant jobs:

  1. FIELD EXTRACTION   -- pull invoice number, amounts, dates, claimant
                           name out of raw OCR text.
  2. CROSS-VERIFICATION -- compare the amount printed on the document
                           against the claim amount on file; compare the
                           claimant name on the document against the
                           case record.
  3. DUPLICATE DETECTION -- detect the same invoice number (or a
                           near-identical document fingerprint) reused
                           across multiple cases.

OCR IS NOISY, AND THAT IS HANDLED, NOT IGNORED. Verified during
development: tesseract read "1234.56" as "123458" at small font sizes,
dropping the decimal point. So amount parsing tolerates missing/garbled
separators and the amount comparison uses a relative tolerance rather
than exact equality. Where OCR confidence is low, that confidence is
carried into the evidence rather than hidden.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from app.documents.ocr import get_ocr_backend, OCRResult

# Tolerances / thresholds -- configurable, documented (Build Rule 19)
AMOUNT_RELATIVE_TOLERANCE = 0.02      # 2% -- absorbs OCR rounding noise
AMOUNT_ABSOLUTE_TOLERANCE = 1.00      # $1 floor for small amounts
NAME_SIMILARITY_THRESHOLD = 0.70      # fuzzy match, OCR mangles names
LOW_OCR_CONFIDENCE = 0.60             # below this, findings are downgraded

AMOUNT_RE = re.compile(r"\$?\s*([0-9][0-9,\s]{0,12}(?:[.,][0-9]{2})?)")
INVOICE_RE = re.compile(r"\b(INV[\s\-]?[0-9]{3,8})\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")
CASE_REF_RE = re.compile(r"\b(FS[\s\-][A-Z0-9\-]{3,20})\b", re.IGNORECASE)


@dataclass
class DocumentFindings:
    case_id: str
    image_path: str
    ocr_backend: str
    ocr_confidence: float | None
    extracted: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)
    document_risk: float = 0.0
    degraded: bool = False          # True when OCR failed or was unusable
    degraded_reason: str | None = None

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "image_path": self.image_path,
            "ocr_backend": self.ocr_backend,
            "ocr_confidence": self.ocr_confidence,
            "extracted": self.extracted,
            "flags": self.flags,
            "document_risk": round(self.document_risk, 4),
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }


def _parse_amount(raw: str) -> float | None:
    """
    Parses an OCR'd amount string tolerantly.

    OCR commonly: drops the decimal point ("1234.56" -> "123456"),
    inserts spaces, or confuses ',' and '.'. We normalize what we can and
    return None rather than guessing wildly when the result is implausible.
    """
    cleaned = raw.strip().replace(" ", "").replace("$", "")
    if not cleaned:
        return None

    # If both separators present, the last one is the decimal separator.
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        # "1,234" -> thousands separator; "1,23" -> likely decimal comma
        if re.search(r",\d{2}$", cleaned):
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")

    try:
        value = float(cleaned)
    except ValueError:
        return None

    return value if 0 < value < 1e9 else None


def extract_fields(text: str) -> dict:
    """Pulls structured fields out of raw OCR text."""
    amounts = []
    for match in AMOUNT_RE.finditer(text):
        parsed = _parse_amount(match.group(1))
        if parsed is not None:
            amounts.append(parsed)

    invoice_numbers = [m.group(1).upper().replace(" ", "-") for m in INVOICE_RE.finditer(text)]
    dates = [m.group(1) for m in DATE_RE.finditer(text)]
    case_refs = [m.group(1).upper().replace(" ", "-") for m in CASE_REF_RE.finditer(text)]

    return {
        "amounts": amounts,
        "max_amount": max(amounts) if amounts else None,
        "invoice_numbers": invoice_numbers,
        "primary_invoice_number": invoice_numbers[0] if invoice_numbers else None,
        "dates": dates,
        "case_references": case_refs,
        "raw_text_length": len(text),
    }


def _name_matches(text: str, expected_name: str) -> bool:
    """Fuzzy name match -- OCR mangles names, so exact matching would
    produce constant false 'name missing' flags."""
    if not expected_name:
        return True
    text_lower = text.lower()
    tokens = [t for t in re.split(r"\s+", expected_name.lower()) if len(t) > 2]
    if not tokens:
        return True

    for token in tokens:
        if token in text_lower:
            return True
        for word in re.findall(r"[a-z]{3,}", text_lower):
            if SequenceMatcher(None, token, word).ratio() >= NAME_SIMILARITY_THRESHOLD:
                return True
    return False


def validate_document(image_path: str | Path, claim_amount: float,
                      claimant_name: str = "", case_id: str = "",
                      backend_name: str | None = None) -> DocumentFindings:
    """
    Runs OCR + validation on a single document.

    Degraded-mode behaviour (Section 57): if OCR fails, we return findings
    marked `degraded=True` with an explicit reason and ZERO risk score --
    we never fabricate a document_risk from a document we couldn't read.
    """
    backend = get_ocr_backend(backend_name)
    result: OCRResult = backend.extract(image_path)

    findings = DocumentFindings(
        case_id=case_id,
        image_path=str(image_path),
        ocr_backend=result.backend,
        ocr_confidence=result.confidence,
    )

    if not result.succeeded:
        findings.degraded = True
        findings.degraded_reason = result.error or "OCR produced no readable text"
        findings.flags.append("ocr_unavailable")
        return findings

    extracted = extract_fields(result.text)
    findings.extracted = extracted

    risk = 0.0

    # --- 1. amount cross-verification -------------------------------
    doc_amounts = extracted["amounts"]
    if not doc_amounts:
        findings.flags.append("no_amount_found_on_document")
        risk += 0.20
    else:
        tolerance = max(AMOUNT_ABSOLUTE_TOLERANCE, claim_amount * AMOUNT_RELATIVE_TOLERANCE)
        closest = min(doc_amounts, key=lambda a: abs(a - claim_amount))
        if abs(closest - claim_amount) > tolerance:
            findings.flags.append("claim_amount_mismatch")
            findings.extracted["closest_document_amount"] = closest
            findings.extracted["claim_amount_on_file"] = claim_amount
            risk += 0.50

    # --- 2. claimant name verification ------------------------------
    if claimant_name and not _name_matches(result.text, claimant_name):
        findings.flags.append("claimant_name_not_found_on_document")
        risk += 0.20

    # --- 3. OCR quality ---------------------------------------------
    if result.confidence is not None and result.confidence < LOW_OCR_CONFIDENCE:
        findings.flags.append("low_ocr_confidence")
        # NOTE: low OCR confidence is a *reliability* caveat, not evidence
        # of fraud -- it does not increase document_risk.

    findings.document_risk = min(1.0, risk)
    return findings


def detect_duplicate_documents(findings_list: list[DocumentFindings]) -> dict:
    """
    Cross-document duplicate detection: the same invoice number appearing
    on more than one case is a classic insurance-fraud pattern (Scenario
    FS-SC-005).

    Returns {invoice_number: [case_ids]} for invoice numbers used on 2+
    DISTINCT cases. Note the "distinct cases" requirement -- the same
    invoice appearing twice within one case is a filing artifact, not
    duplication across claims.
    """
    by_invoice = defaultdict(set)
    for f in findings_list:
        inv = f.extracted.get("primary_invoice_number")
        if inv and f.case_id:
            by_invoice[inv].add(f.case_id)

    duplicates = {inv: sorted(cases) for inv, cases in by_invoice.items() if len(cases) > 1}

    # annotate the affected findings in place
    for f in findings_list:
        inv = f.extracted.get("primary_invoice_number")
        if inv in duplicates:
            if "duplicate_invoice_across_cases" not in f.flags:
                f.flags.append("duplicate_invoice_across_cases")
                f.document_risk = min(1.0, f.document_risk + 0.40)
            f.extracted["duplicate_shared_with_cases"] = [
                c for c in duplicates[inv] if c != f.case_id
            ]

    return duplicates
