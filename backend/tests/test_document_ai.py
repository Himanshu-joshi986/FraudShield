"""
Phase 6 tests -- Document AI.

Unlike Phases 4/5, this phase ran END TO END in the build environment:
tesseract 5.3.4 is installed, so real OCR was performed on real generated
images. These tests exercise the actual pipeline, not stubs.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO_ROOT / "data" / "documents"
METRICS_PATH = REPO_ROOT / "docs" / "document_ai_metrics.json"

import sys
sys.path.insert(0, str(REPO_ROOT / "backend"))


class TestAmountParsing:
    """Unit tests for the OCR-noise-tolerant amount parser."""

    def test_parses_plain_amounts(self):
        from app.documents.validator import _parse_amount
        assert _parse_amount("1234.56") == 1234.56
        assert _parse_amount("$ 5,000.00") == 5000.00
        assert _parse_amount("  7500  ") == 7500.0

    def test_handles_thousands_separators(self):
        from app.documents.validator import _parse_amount
        assert _parse_amount("1,234") == 1234.0
        assert _parse_amount("12,345.67") == 12345.67

    def test_handles_european_decimal_comma(self):
        from app.documents.validator import _parse_amount
        assert _parse_amount("1234,56") == 1234.56

    def test_rejects_garbage(self):
        from app.documents.validator import _parse_amount
        assert _parse_amount("") is None
        assert _parse_amount("abc") is None
        assert _parse_amount("0") is None


class TestFieldExtraction:
    def test_extracts_invoice_number_and_amount(self):
        from app.documents.validator import extract_fields
        text = "INVOICE\nInvoice Number: INV-12345\nTOTAL AMOUNT: $ 7,500.00\nDate: 2026-05-14"
        fields = extract_fields(text)
        assert fields["primary_invoice_number"] == "INV-12345"
        assert 7500.0 in fields["amounts"]
        assert "2026-05-14" in fields["dates"]

    def test_handles_text_with_no_fields(self):
        from app.documents.validator import extract_fields
        fields = extract_fields("nothing useful here")
        assert fields["primary_invoice_number"] is None
        assert fields["amounts"] == []


class TestOCRBackendSelection:
    def test_defaults_to_tesseract(self):
        from app.documents.ocr import get_ocr_backend
        assert get_ocr_backend().name == "tesseract"

    def test_unknown_backend_raises_rather_than_silently_defaulting(self):
        """A typo must not silently produce results from a different engine
        than the one the user thinks they selected."""
        from app.documents.ocr import get_ocr_backend
        with pytest.raises(ValueError):
            get_ocr_backend("not_a_real_engine")

    def test_paddleocr_raises_clear_error_when_not_installed(self):
        """Section 57: never silently fall back. Selecting PaddleOCR
        without installing it must fail loudly with install instructions."""
        from app.documents.ocr import PaddleOCRBackend
        backend = PaddleOCRBackend()
        result = backend.extract("/nonexistent.png")
        assert not result.succeeded
        assert result.error is not None


@pytest.mark.skipif(not (DOCS_DIR / "manifest.json").exists(),
                    reason="run backend/app/documents/generator.py first")
class TestGeneratedDocuments:
    @classmethod
    def setup_class(cls):
        with open(DOCS_DIR / "manifest.json") as f:
            cls.manifest = json.load(f)

    def test_documents_were_generated(self):
        assert len(self.manifest) > 0
        for entry in self.manifest:
            assert (REPO_ROOT / entry["image_path"]).exists()

    def test_all_documents_marked_synthetic(self):
        for entry in self.manifest:
            assert entry["data_source"] == "synthetic"
            assert "SYNTHETIC" in Path(entry["image_path"]).name

    def test_ground_truth_contains_planted_anomalies(self):
        """The evaluation is only meaningful if anomalies were actually
        planted -- verify the generator did its job."""
        mismatches = [e for e in self.manifest if e["expected_amount_mismatch"]]
        duplicates = [e for e in self.manifest if e["expected_duplicate_of"]]
        assert len(mismatches) >= 1
        assert len(duplicates) >= 1

    def test_duplicate_flag_is_not_universally_true(self):
        """Regression guard for a real bug found during development:
        pandas reads empty CSV cells as NaN, and str(nan) == 'nan' which is
        truthy -- which marked every single document a duplicate."""
        non_duplicates = [e for e in self.manifest if not e["expected_duplicate_of"]]
        assert len(non_duplicates) > 0, "every document marked duplicate -- NaN bug regression"


@pytest.mark.skipif(not METRICS_PATH.exists(),
                    reason="run scripts/evaluate_document_ai.py first")
class TestDocumentAIEvaluation:
    @classmethod
    def setup_class(cls):
        with open(METRICS_PATH) as f:
            cls.metrics = json.load(f)

    def test_ocr_actually_ran(self):
        assert self.metrics["ocr_backend"] == "tesseract"
        assert self.metrics["ocr_failures"] == 0

    def test_ocr_confidence_is_reasonable(self):
        assert self.metrics["mean_ocr_confidence"] > 0.7

    def test_invoice_numbers_extracted_correctly(self):
        assert self.metrics["invoice_number_extraction_accuracy"] >= 0.9

    def test_amount_mismatches_detected(self):
        m = self.metrics["amount_mismatch_detection"]
        assert m["recall"] == 1.0, "failed to catch a planted amount mismatch"
        assert m["fn"] == 0

    def test_duplicates_detected_with_no_misses(self):
        strict = self.metrics["duplicate_detection_strict_labeling"]
        assert strict["recall"] == 1.0
        assert strict["fn"] == 0

    def test_both_labeling_framings_are_reported(self):
        """Honesty guard: the project reports both the strict framing
        (which looks worse) and the operational framing, rather than
        silently picking whichever number flatters the system."""
        assert "duplicate_detection_strict_labeling" in self.metrics
        assert "duplicate_detection_implicated_labeling" in self.metrics
        assert "note_on_the_two_framings" in self.metrics

    def test_scope_caveat_present(self):
        """9 synthetic documents is a tiny evaluation set; the metrics file
        must say so rather than presenting these as general accuracy."""
        assert "scope_caveat" in self.metrics
        assert "NOT a claim" in self.metrics["scope_caveat"]


class TestDegradedMode:
    def test_unreadable_document_returns_degraded_not_fabricated_risk(self):
        """
        Section 57: 'Never silently fabricate an output.' If OCR can't read
        a document, document_risk must be 0.0 and degraded must be True --
        not an invented risk score derived from nothing.
        """
        from app.documents.validator import validate_document
        findings = validate_document(
            image_path="/nonexistent/file.png",
            claim_amount=5000.0,
            claimant_name="Test Person",
            case_id="FS-TEST-001",
        )
        assert findings.degraded is True
        assert findings.degraded_reason is not None
        assert findings.document_risk == 0.0
        assert "ocr_unavailable" in findings.flags
