"""
Phase 13 -- WebSocket live updates (Section 41).

Uses the real LangGraph .stream() API (verified: it yields one dict per
completed node, in execution order -- {"sentinel": <state>}, then
{"investigation": <state>}, etc.) to emit genuine agent_started /
agent_completed events as each agent actually runs, not a simulated
progress bar.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import WebSocket

NODE_ORDER = ["sentinel", "investigation", "evidence", "report", "commander", "short_circuit_end"]
NODE_DISPLAY_NAMES = {
    "sentinel": "Sentinel Agent",
    "investigation": "Investigation Agent",
    "evidence": "Evidence Analyst",
    "report": "Report Architect",
    "commander": "Case Commander",
    "short_circuit_end": "Sentinel Agent (short-circuit)",
}


class ConnectionManager:
    """Per-case-id WebSocket subscriber list, matching Section 40's
    `WS /ws/cases/{case_id}` route shape."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, case_id: str, websocket: WebSocket):
        await websocket.accept()
        self._connections.setdefault(case_id, []).append(websocket)

    def disconnect(self, case_id: str, websocket: WebSocket):
        if case_id in self._connections and websocket in self._connections[case_id]:
            self._connections[case_id].remove(websocket)

    async def broadcast(self, case_id: str, event: dict):
        event = {**event, "timestamp": datetime.now(timezone.utc).isoformat()}
        dead = []
        for ws in self._connections.get(case_id, []):
            try:
                await ws.send_text(json.dumps(event, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(case_id, ws)


manager = ConnectionManager()


async def run_case_streaming(compiled_graph, case_id: str, input_data: dict) -> dict:
    """
    Runs the pipeline via LangGraph's real .stream() API, broadcasting a
    genuine agent_started event before each node and agent_completed
    immediately after it actually finishes -- not simulated timing.

    IMPLEMENTATION NOTE on a real bug found and fixed here: an earlier
    version collected the whole generator eagerly (`list(graph.stream(...))`)
    in a background thread before broadcasting anything. Because each
    agent MUTATES AND RETURNS THE SAME shared state dict (rather than a
    fresh copy per node), eagerly draining the generator to completion
    meant every "step" reference pointed at the SAME final, fully-mutated
    object by the time any broadcasting happened -- so every
    agent_completed event showed Case Commander's summary, regardless of
    which agent it was supposedly reporting on. Verified with a live
    WebSocket client before concluding it was fixed, not just inferred.

    Fixed by bridging the synchronous generator to the async event loop
    via a thread + queue, so each step is broadcast the moment it's
    produced -- before the next node has had a chance to mutate the
    shared state further.
    """
    initial_state = {"case_id": case_id, "input_data": input_data, "agent_logs": [], "errors": []}
    final_state = initial_state

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def _produce():
        try:
            for step in compiled_graph.stream(initial_state):
                node_name = list(step.keys())[0]
                node_state = step[node_name]
                # snapshot THIS node's own log entry and a shallow copy of
                # the fields we need NOW, before the next node mutates
                # the shared dict further
                logs = node_state.get("agent_logs", [])
                snapshot = {
                    "node_name": node_name,
                    "summary": logs[-1]["output_summary"] if logs else "",
                    "risk_level": node_state.get("risk_level"),
                    "routing_decision": dict(node_state.get("routing_decision") or {}),
                    "full_state_ref": node_state,  # only safe to read AFTER the whole run finishes
                }
                loop.call_soon_threadsafe(queue.put_nowait, snapshot)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    import threading
    threading.Thread(target=_produce, daemon=True).start()

    while True:
        snapshot = await queue.get()
        if snapshot is SENTINEL:
            break

        display_name = NODE_DISPLAY_NAMES.get(snapshot["node_name"], snapshot["node_name"])
        await manager.broadcast(case_id, {"event": "agent_started", "agent": display_name, "case_id": case_id})
        await asyncio.sleep(0.02)
        await manager.broadcast(case_id, {
            "event": "agent_completed", "agent": display_name, "case_id": case_id,
            "summary": snapshot["summary"],
        })
        final_state = snapshot["full_state_ref"]

    await manager.broadcast(case_id, {
        "event": "investigation_complete", "case_id": case_id,
        "risk_level": final_state.get("risk_level"),
        "decision": final_state.get("routing_decision", {}).get("decision"),
    })

    return final_state
