"""
Phase 13 tests -- WebSocket live updates.

The most important test here (test_streaming_events_are_distinct_per_agent)
is a regression guard for a real bug found and fixed during development:
an earlier implementation eagerly drained LangGraph's .stream() generator
in a background thread before broadcasting anything. Because each agent
mutates and returns the SAME shared state dict, every "step" reference
ended up pointing at the final, fully-mutated object by the time any
event was sent -- so every agent_completed event showed Case Commander's
summary, regardless of which agent it claimed to report on. Verified
live with a real websockets client before concluding it was fixed.
"""
import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

GRAPH_DB_PATH = REPO_ROOT / "data" / "graph_db"
CHROMA_PATH = REPO_ROOT / "data" / "chroma_db"

pytestmark = pytest.mark.skipif(
    not GRAPH_DB_PATH.exists() or not CHROMA_PATH.exists(),
    reason="run scripts/ingest_graph.py and scripts/evaluate_rag.py first",
)


class FakeWebSocket:
    def __init__(self, sink):
        self.sink = sink

    async def accept(self):
        pass

    async def send_text(self, text):
        self.sink.append(text)


class TestConnectionManager:
    def test_broadcast_to_no_subscribers_does_not_raise(self):
        from app.services.websocket_manager import ConnectionManager
        manager = ConnectionManager()
        asyncio.run(manager.broadcast("FS-NOBODY-LISTENING", {"event": "test"}))

    def test_disconnect_unknown_case_does_not_raise(self):
        from app.services.websocket_manager import ConnectionManager
        manager = ConnectionManager()
        manager.disconnect("FS-UNKNOWN", object())


class TestStreamingPipeline:
    """Runs the REAL pipeline via run_case_streaming against the real
    graph store / knowledge base, with a fake WebSocket collecting every
    broadcast event -- the actual code path the live server uses."""

    @pytest.fixture
    def compiled_graph_and_case(self):
        import pandas as pd
        from app.agents.graph_pipeline import build_graph
        from app.graph.store import get_graph_store
        from app.rag.knowledge_base import get_knowledge_base

        gs = get_graph_store()
        kb = get_knowledge_base()
        cases_df = pd.read_csv(REPO_ROOT / "data" / "synthetic" / "synthetic_cases.csv")
        ring_row = cases_df[cases_df["scenario_id"] == "FS-SC-003"].iloc[0]

        compiled = build_graph(gs, kb, {}, all_cases=cases_df.to_dict("records"))
        input_data = {
            "customer_id": ring_row["customer_id"], "customer_name": ring_row["customer_name"],
            "amount": float(ring_row["amount"]), "domain": ring_row["domain"],
            "timestamp": ring_row["timestamp"], "scenario_id": ring_row["scenario_id"],
        }
        yield compiled, ring_row["case_id"], input_data
        gs.close()

    def _run_and_collect(self, compiled, case_id, input_data):
        from app.services.websocket_manager import run_case_streaming, manager
        collected = []

        async def run():
            ws = FakeWebSocket(collected)
            manager._connections[case_id] = [ws]
            return await run_case_streaming(compiled, case_id, input_data)

        final_state = asyncio.run(run())
        import json
        events = [json.loads(e) for e in collected]
        return events, final_state

    def test_streaming_events_are_distinct_per_agent(self, compiled_graph_and_case):
        """THE regression test for the shared-mutable-state bug described
        in this file's docstring."""
        compiled, case_id, input_data = compiled_graph_and_case
        events, _ = self._run_and_collect(compiled, case_id, input_data)

        completed = [e for e in events if e["event"] == "agent_completed"]
        assert len(completed) == 5
        summaries = [e["summary"] for e in completed]
        assert len(set(summaries)) == 5, (
            f"agent summaries were not distinct -- regression of the shared-mutable-state bug: {summaries}"
        )

    def test_events_arrive_in_correct_agent_order(self, compiled_graph_and_case):
        compiled, case_id, input_data = compiled_graph_and_case
        events, _ = self._run_and_collect(compiled, case_id, input_data)

        completed_agents = [e["agent"] for e in events if e["event"] == "agent_completed"]
        assert completed_agents == [
            "Sentinel Agent", "Investigation Agent", "Evidence Analyst",
            "Report Architect", "Case Commander",
        ]

    def test_agent_started_precedes_its_own_completed(self, compiled_graph_and_case):
        compiled, case_id, input_data = compiled_graph_and_case
        events, _ = self._run_and_collect(compiled, case_id, input_data)

        pipeline_events = [e for e in events if e["event"] in ("agent_started", "agent_completed")]
        for i in range(0, len(pipeline_events), 2):
            assert pipeline_events[i]["event"] == "agent_started"
            assert pipeline_events[i + 1]["event"] == "agent_completed"
            assert pipeline_events[i]["agent"] == pipeline_events[i + 1]["agent"]

    def test_final_investigation_complete_event_sent(self, compiled_graph_and_case):
        compiled, case_id, input_data = compiled_graph_and_case
        events, final_state = self._run_and_collect(compiled, case_id, input_data)

        assert events[-1]["event"] == "investigation_complete"
        assert events[-1]["decision"] == final_state["routing_decision"]["decision"]

    def test_return_value_matches_non_streaming_pipeline_contract(self, compiled_graph_and_case):
        """The streaming path must return the same shape run_case() does,
        so callers (pipeline_service.investigate_case) don't need to
        special-case it."""
        compiled, case_id, input_data = compiled_graph_and_case
        _, final_state = self._run_and_collect(compiled, case_id, input_data)

        for key in ["case_id", "risk_score", "risk_level", "evidence", "routing_decision", "agent_logs"]:
            assert key in final_state
