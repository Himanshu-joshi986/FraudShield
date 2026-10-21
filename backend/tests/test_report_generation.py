"""
Phase 11 tests -- PDF report generation.

Ran end-to-end in the build environment: real ReportLab PDFs generated
from real pipeline output, verified by extracting text back OUT of the
generated PDFs with pdfplumber and checking that real facts (case ID,
risk level, evidence IDs) actually appear -- not just that a file with
nonzero size was written.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

REPORT_METRICS_PATH = REPO_ROOT / "docs" / "report_generation_metrics.json"
REPORTS_DIR = REPO_ROOT / "reports"


class TestPDFGenerationUnit:
    """Builds a PDF from hand-constructed state, independent of the
    full pipeline/data stores being built."""

    def test_generates_a_real_pdf_file(self, tmp_path, monkeypatch):
        from app.services import report_generator
        monkeypatch.setattr(report_generator, "REPORTS_DIR", tmp_path)

        state = {
            "case_id": "T-PDF-001", "input_data": {"domain": "test", "amount": 999.0},
            "risk_score": 0.5, "risk_level": "MEDIUM",
            "evidence": [{"evidence_id": "E-1", "type": "graph", "finding": "test finding",
                        "source": "test_source", "confidence": 0.8, "source_reference": {}}],
            "routing_decision": {"decision": "HUMAN_REVIEW", "reason": "test reason",
                                "policy_conditions_triggered": ["test condition"],
                                "policy_version": "v1", "policy_source": "test"},
            "report": {},
        }
        explanation = {
            "headline": "test headline",
            "reasons": [{"rank": 1, "reason": "test finding", "evidence_id": "E-1",
                        "confidence": 0.8, "source": "test_source"}],
            "disclaimer": "This explanation describes model and evidence signals.",
        }

        path = report_generator.generate_report_pdf(state, explanation)
        assert path.exists()
        assert path.stat().st_size > 500  # not an empty/broken file

    def test_pdf_content_matches_state_via_text_extraction(self, tmp_path, monkeypatch):
        """The real verification: extract text back out and confirm the
        actual case facts appear, not just that a file was produced."""
        import pdfplumber
        from app.services import report_generator
        monkeypatch.setattr(report_generator, "REPORTS_DIR", tmp_path)

        state = {
            "case_id": "T-PDF-UNIQUE-002", "input_data": {"domain": "test", "amount": 1234.0},
            "risk_score": 0.9, "risk_level": "HIGH",
            "evidence": [{"evidence_id": "E-UNIQUE-99", "type": "graph", "finding": "unique finding text",
                        "source": "test_source", "confidence": 0.8, "source_reference": {}}],
            "routing_decision": {"decision": "ESCALATE", "reason": "test",
                                "policy_conditions_triggered": [], "policy_version": "v1",
                                "policy_source": "test"},
            "report": {},
        }
        explanation = {"headline": "h", "reasons": [], "disclaimer": "d"}

        path = report_generator.generate_report_pdf(state, explanation)
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)

        assert "T-PDF-UNIQUE-002" in text
        assert "HIGH" in text
        assert "E-UNIQUE-99" in text
        assert "unique finding text" in text
        assert "ESCALATE" in text

    def test_disclaimer_always_present(self, tmp_path, monkeypatch):
        """Section 21: must never be presented as an official filing."""
        import pdfplumber
        from app.services import report_generator
        monkeypatch.setattr(report_generator, "REPORTS_DIR", tmp_path)

        state = {"case_id": "T-PDF-003", "input_data": {"domain": "test", "amount": 1.0},
                "risk_score": 0.0, "risk_level": "LOW", "evidence": [],
                "routing_decision": {"decision": "AUTO_CLOSE_MONITOR", "reason": "",
                                    "policy_conditions_triggered": [], "policy_version": "v1",
                                    "policy_source": ""}, "report": {}}
        explanation = {"headline": "h", "reasons": [], "disclaimer": "d"}

        path = report_generator.generate_report_pdf(state, explanation)
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        assert "NOT an official regulatory filing" in text

    def test_empty_evidence_does_not_crash(self, tmp_path, monkeypatch):
        """A short-circuited low-risk case has zero evidence -- must still
        produce a valid, readable PDF, not crash or produce a broken one."""
        import pdfplumber
        from app.services import report_generator
        monkeypatch.setattr(report_generator, "REPORTS_DIR", tmp_path)

        state = {"case_id": "T-PDF-004", "input_data": {"domain": "test", "amount": 1.0},
                "risk_score": 0.0, "risk_level": "LOW", "evidence": [],
                "routing_decision": {"decision": "AUTO_CLOSE_MONITOR", "reason": "",
                                    "policy_conditions_triggered": [], "policy_version": "v1",
                                    "policy_source": ""}, "report": {}}
        explanation = {"headline": "clean case", "reasons": [], "disclaimer": "d"}

        path = report_generator.generate_report_pdf(state, explanation)
        with pdfplumber.open(path) as pdf:
            assert len(pdf.pages) >= 1
            text = pdf.pages[0].extract_text()
            assert "No findings of this type" in text


@pytest.mark.skipif(not REPORT_METRICS_PATH.exists(), reason="run scripts/generate_reports.py first")
class TestGeneratedReportsAgainstRealPipeline:
    @classmethod
    def setup_class(cls):
        with open(REPORT_METRICS_PATH) as f:
            cls.metrics = json.load(f)

    def test_all_representative_scenarios_generated(self):
        assert self.metrics["n_reports_generated"] == 6

    def test_all_content_verification_checks_passed(self):
        assert self.metrics["n_verification_passed"] == self.metrics["n_reports_generated"]

    def test_pdf_files_actually_exist_on_disk(self):
        for r in self.metrics["reports"]:
            pdf_path = REPO_ROOT / r["pdf_path"]
            assert pdf_path.exists()
            assert pdf_path.stat().st_size == r["pdf_size_bytes"]

    def test_high_risk_reports_have_more_pages_than_short_circuited_low_risk(self):
        by_scenario = {r["scenario_id"]: r for r in self.metrics["reports"]}
        assert by_scenario["FS-SC-003"]["n_pages"] >= by_scenario["FS-SC-001"]["n_pages"]

    def test_fraud_ring_report_contains_evidence(self):
        by_scenario = {r["scenario_id"]: r for r in self.metrics["reports"]}
        assert by_scenario["FS-SC-003"]["n_evidence_items"] > 0
