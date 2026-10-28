"""
Phase 9 tests -- the 5-agent LangGraph pipeline.

Ran end-to-end in the build environment against real tool calls (kuzu
graph queries, ChromaDB RAG queries, document OCR). The Gemini LLM calls
are unreachable here (verified: 403 Forbidden) and exercise the
deterministic-fallback path for real -- this is genuine coverage of the
Section 57 error-handling requirement under real failure conditions, not
a simulated failure.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

PIPELINE_METRICS_PATH = REPO_ROOT / "docs" / "agent_pipeline_metrics.json"


class TestRiskFusion:
    def test_all_signals_available(self):
        from app.ml.fusion.risk_fusion import fuse
        result = fuse({"tabular_ml_risk": 0.8, "graph_risk": 0.8, "temporal_risk": 0.8, "document_risk": 0.8})
        assert result["risk_score"] == pytest.approx(0.8, abs=0.01)
        assert result["risk_level"] == "HIGH"
        assert not result["degraded"]

    def test_missing_signals_are_excluded_not_zeroed(self):
        from app.ml.fusion.risk_fusion import fuse
        result = fuse({"tabular_ml_risk": None, "graph_risk": 0.8, "temporal_risk": None, "document_risk": None})
        assert result["risk_score"] == pytest.approx(0.8, abs=0.01)
        assert result["available_signals"] == ["graph_risk"]
        assert set(result["missing_signals"]) == {"tabular_ml_risk", "temporal_risk", "document_risk"}

    def test_no_signals_available_returns_degraded_not_fabricated(self):
        from app.ml.fusion.risk_fusion import fuse
        result = fuse({"tabular_ml_risk": None, "graph_risk": None, "temporal_risk": None, "document_risk": None})
        assert result["risk_score"] == 0.0
        assert result["risk_level"] == "UNSCORED"
        assert result["degraded"] is True

    def test_no_available_signals_requires_human_review(self):
        from app.agents.case_commander import run
        state = {
            "case_id": "T-NO-SIGNALS",
            "risk_score": 0.0,
            "n_override_eligible_connections": 0,
            "signals": [],
            "model_results": {"fusion": {"available_signals": [], "degraded": True}},
        }
        result = run(state)
        assert result["routing_decision"]["decision"] == "HUMAN_REVIEW"
        assert "no risk signals available" in result["routing_decision"]["reason"]

    def test_risk_level_bands(self):
        from app.ml.fusion.risk_fusion import fuse
        low = fuse({"graph_risk": 0.1})
        medium = fuse({"graph_risk": 0.5})
        high = fuse({"graph_risk": 0.9})
        assert low["risk_level"] == "LOW"
        assert medium["risk_level"] == "MEDIUM"
        assert high["risk_level"] == "HIGH"


class TestAgentStateSchema:
    """
    Regression guard for a real bug found while building the case insights.

    LangGraph builds its state channels from the FraudCaseState schema and
    SILENTLY DROPS any node update for a key that is not declared there.
    Sentinel writes `n_override_eligible_connections` and Case Commander
    reads it to apply the coordinated-ring override -- but the key was
    missing from the schema, so the commander always saw 0 connections and
    the ring override never fired. Every ring case was routed on its score
    alone, and the policy trace showed no override on cases that clearly
    had one.
    """

    def test_ring_override_key_is_declared_in_the_state_schema(self):
        from app.agents.state import FraudCaseState
        assert "n_override_eligible_connections" in FraudCaseState.__annotations__

    def test_every_state_key_written_by_an_agent_is_declared(self):
        """Guards against the same class of bug returning in another key."""
        import re
        from app.agents.state import FraudCaseState

        agents_dir = REPO_ROOT / "backend" / "app" / "agents"
        declared = set(FraudCaseState.__annotations__)
        written = set()
        for path in agents_dir.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            written.update(re.findall(r'state\[["\']([a-z_]+)["\']\]\s*=', source))
            written.update(
                m for m in re.findall(r'state\.setdefault\(["\']([a-z_]+)["\']', source)
            )

        # agent_logs / errors are appended to, not replaced, but they are
        # declared too -- so every key written must be in the schema.
        undeclared = written - declared
        assert not undeclared, f"state keys written but not declared: {sorted(undeclared)}"

    def test_ring_override_fires_when_connections_reach_the_threshold(self):
        from app.agents.case_commander import run
        state = {
            "case_id": "T-RING",
            "risk_score": 0.10,  # well below the HIGH threshold
            "n_override_eligible_connections": 4,
            "signals": [],
            "model_results": {"fusion": {"available_signals": ["graph_risk"], "degraded": False}},
        }
        result = run(state)
        assert result["routing_decision"]["decision"] == "ESCALATE"
        assert "ring_override" in result["routing_decision"]["reason"]


class TestLLMClientDegradation:
    def test_missing_api_key_degrades_cleanly(self):
        from app.agents.llm_client import GeminiClient
        client = GeminiClient(api_key="")
        result = client.complete("test prompt")
        assert result.succeeded is False
        assert "not configured" in result.degraded_reason

    def test_request_budget_enforces_daily_limit(self):
        from app.agents.llm_client import RequestBudget
        budget = RequestBudget(max_per_minute=100, max_per_day=2)
        assert budget.can_proceed() == (True, None)
        budget.record()
        assert budget.can_proceed() == (True, None)
        budget.record()
        can, reason = budget.can_proceed()
        assert can is False
        assert "daily" in reason

    def test_request_budget_enforces_per_minute_limit(self):
        from app.agents.llm_client import RequestBudget
        budget = RequestBudget(max_per_minute=2, max_per_day=1000)
        budget.record()
        budget.record()
        can, reason = budget.can_proceed()
        assert can is False
        assert "per-minute" in reason

    def test_degraded_result_never_raises(self):
        from app.agents.llm_client import GeminiClient
        client = GeminiClient(api_key="fake-key-that-will-fail", max_per_day=1000, max_per_minute=1000)
        result = client.complete("test")
        assert result.succeeded is False


class TestIndividualAgents:
    def test_case_commander_ring_override(self):
        from app.agents.case_commander import run
        state = {"case_id": "T1", "risk_score": 0.1, "n_override_eligible_connections": 3, "signals": []}
        result = run(state)
        assert result["routing_decision"]["decision"] == "ESCALATE"
        assert "ring_override" in result["routing_decision"]["reason"]

    def test_case_commander_address_only_does_not_trigger_ring_override(self):
        """Regression guard for the real bug found during Phase 9
        evaluation: Case Commander must use the override-eligible
        connection count (Account/Device), not the raw graph_findings
        connection count (which includes Address)."""
        from app.agents.case_commander import run
        state = {
            "case_id": "T2", "risk_score": 0.1, "n_override_eligible_connections": 0,
            "signals": [], "graph_findings": {"n_connections": 4},
        }
        result = run(state)
        assert result["routing_decision"]["decision"] != "ESCALATE"

    def test_case_commander_document_mismatch_override(self):
        from app.agents.case_commander import run
        state = {"case_id": "T3", "risk_score": 0.05, "n_override_eligible_connections": 0,
                "signals": ["claim_amount_mismatch"]}
        result = run(state)
        assert result["routing_decision"]["decision"] == "HUMAN_REVIEW"

    def test_case_commander_velocity_override(self):
        from app.agents.case_commander import run
        state = {"case_id": "T4", "risk_score": 0.05, "n_override_eligible_connections": 0,
                "signals": ["high_transaction_velocity"]}
        result = run(state)
        assert result["routing_decision"]["decision"] == "HUMAN_REVIEW"

    def test_case_commander_low_risk_auto_closes(self):
        from app.agents.case_commander import run
        state = {"case_id": "T5", "risk_score": 0.05, "n_override_eligible_connections": 0, "signals": []}
        result = run(state)
        assert result["routing_decision"]["decision"] == "AUTO_CLOSE_MONITOR"

    def test_case_commander_records_triggering_conditions(self):
        from app.agents.case_commander import run
        state = {"case_id": "T6", "risk_score": 0.9, "n_override_eligible_connections": 0, "signals": []}
        result = run(state)
        assert len(result["routing_decision"]["policy_conditions_triggered"]) > 0

    def test_evidence_analyst_assigns_unique_ids(self):
        from app.agents.evidence_agent import run
        state = {
            "case_id": "T7",
            "graph_findings": {"n_connections": 2, "connections": []},
            "model_results": {"signals": {"document_risk": 0.5}, "fusion": {}},
            "retrieved_knowledge": [], "investigation_findings": {}, "signals": [],
        }
        result = run(state)
        ids = [e["evidence_id"] for e in result["evidence"]]
        assert len(ids) == len(set(ids))

    def test_evidence_analyst_every_item_has_source_reference(self):
        from app.agents.evidence_agent import run
        state = {
            "case_id": "T8",
            "graph_findings": {"n_connections": 3, "connections": [
                {"connected_customer": "C2", "shared_type": "Account", "shared_entity": "ACC-1"}]},
            "model_results": {"signals": {}, "fusion": {}},
            "retrieved_knowledge": [], "investigation_findings": {}, "signals": [],
        }
        result = run(state)
        for e in result["evidence"]:
            assert "source_reference" in e and e["source_reference"] is not None

    def test_no_signal_evidence_does_not_claim_zero_risk(self):
        from app.agents.evidence_agent import run
        state = {
            "case_id": "T-NO-EVIDENCE",
            "model_results": {"signals": {}, "fusion": {
                "available_signals": [], "missing_signals": ["graph_risk"], "degraded": True,
            }},
            "risk_score": 0.0,
            "retrieved_knowledge": [],
            "investigation_findings": {},
            "signals": [],
        }
        result = run(state)
        model_evidence = next(e for e in result["evidence"] if e["source"].startswith("risk_fusion"))
        assert "Risk score unavailable" in model_evidence["finding"]
        assert model_evidence["confidence"] == 0.0

    def test_report_architect_never_invents_beyond_evidence(self):
        from app.agents.report_agent import run
        state = {"case_id": "T9", "input_data": {"domain": "test", "amount": 100},
                "evidence": [], "risk_score": 0.1, "risk_level": "LOW",
                "signals": [], "model_results": {"fusion": {}}}
        result = run(state)
        assert "no findings of this type" in result["report"]["content"]

    def test_report_architect_includes_disclaimer(self):
        from app.agents.report_agent import run
        state = {"case_id": "T10", "input_data": {"domain": "test", "amount": 100},
                "evidence": [], "risk_score": 0.1, "risk_level": "LOW",
                "signals": [], "model_results": {"fusion": {}}}
        result = run(state)
        assert "NOT an official regulatory filing" in result["report"]["content"]
        assert result["report"]["disclaimer"]


@pytest.mark.skipif(not PIPELINE_METRICS_PATH.exists(), reason="run scripts/run_agent_pipeline.py first")
class TestFullPipelineEvaluation:
    @classmethod
    def setup_class(cls):
        with open(PIPELINE_METRICS_PATH) as f:
            cls.metrics = json.load(f)

    def test_all_cases_completed_without_crashing(self):
        assert self.metrics["workflow_completion_rate"] == 1.0

    def test_llm_degradation_is_real_not_simulated(self):
        assert self.metrics["n_cases_with_llm_degradation"] > 0
        assert "403 Forbidden" in self.metrics["llm_degradation_note"]

    def test_routing_consistency_is_measured_and_reasonable(self):
        """Deliberately does not require 100% -- see the documented
        Scenario 6 gap in docs/EXPERIMENTS.md. Requiring perfection here
        would incentivize curve-fitting weights to this exact eval set."""
        assert self.metrics["routing_consistency_rate"] >= 0.8

    def test_planted_fraud_ring_scenario_fully_correct(self):
        assert self.metrics["per_scenario"]["FS-SC-003"]["all_matched"] is True

    def test_legitimate_case_never_escalated(self):
        assert self.metrics["per_scenario"]["FS-SC-001"]["all_matched"] is True

    def test_low_risk_short_circuit_actually_fired(self):
        assert self.metrics["n_short_circuited"] > 0

    def test_document_mismatch_and_duplicate_scenarios_correct_after_fixes(self):
        assert self.metrics["per_scenario"]["FS-SC-004"]["all_matched"] is True
        assert self.metrics["per_scenario"]["FS-SC-005"]["all_matched"] is True

    def test_known_open_gap_is_documented(self):
        """Scenario 6 (address-only sharing) currently under-scores the
        MEDIUM threshold -- a genuine, documented calibration question,
        not silently ignored."""
        assert self.metrics["per_scenario"]["FS-SC-006"]["all_matched"] is False
