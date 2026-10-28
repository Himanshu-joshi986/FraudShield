from app.agents.sentinel_agent import _velocity_risk


def _case(case_id, timestamp, amount):
    return {
        "case_id": case_id,
        "customer_id": "C-1",
        "timestamp": timestamp,
        "amount": amount,
        "domain": "financial_transaction",
    }


def test_velocity_ignores_future_activity_self_and_same_time_ties():
    current = _case("C-DOC-101", "2026-10-21T09:02:00+00:00", 1840.00)
    all_cases = [
        current,
        _case("FS-DEMO-001", "2026-10-21T09:02:00+00:00", 1840.00),
        _case("C-DOC-104", "2026-10-21T09:44:00+00:00", 1295.75),
    ]

    risk, flags, count = _velocity_risk(current, all_cases, case_id="C-DOC-101")

    assert risk == 0.0
    assert flags == []
    assert count == 1


def test_velocity_counts_distinct_prior_events_inside_trailing_window():
    current = _case("C-3", "2026-10-21T09:27:00+00:00", 3190.00)
    all_cases = [
        _case("C-1", "2026-10-21T09:02:00+00:00", 1840.00),
        _case("DUPLICATE-C-1", "2026-10-21T09:02:00+00:00", 1840.00),
        _case("C-2", "2026-10-21T09:11:00+00:00", 2275.50),
        current,
        _case("C-4", "2026-10-21T09:44:00+00:00", 1295.75),
    ]

    risk, flags, count = _velocity_risk(current, all_cases, case_id="C-3")

    assert risk == 0.25
    assert flags == ["high_transaction_velocity"]
    assert count == 3


def test_velocity_does_not_use_events_older_than_one_hour():
    current = _case("C-2", "2026-10-21T10:03:00+00:00", 100.00)
    all_cases = [
        _case("C-1", "2026-10-21T09:02:00+00:00", 100.00),
        current,
    ]

    risk, flags, count = _velocity_risk(current, all_cases, case_id="C-2")

    assert risk == 0.0
    assert flags == []
    assert count == 1


def test_velocity_override_records_the_observed_window_count():
    from app.agents.case_commander import run

    result = run({
        "case_id": "C-3",
        "risk_score": 0.2,
        "n_override_eligible_connections": 0,
        "signals": ["high_transaction_velocity"],
        "velocity_burst_count": 3,
        "velocity_window_seconds": 3600,
    })

    assert result["routing_decision"]["decision"] == "HUMAN_REVIEW"
    assert "3 distinct customer events within the trailing 60 minutes" in result["routing_decision"]["reason"]


def test_velocity_review_floor_is_medium_and_preserves_the_signal_average():
    from app.agents.sentinel_agent import run

    class EmptyGraph:
        def find_connected_customers(self, _customer_id):
            return []

    current = _case("C-3", "2026-10-21T09:27:00+00:00", 3190.00)
    all_cases = [
        _case("C-1", "2026-10-21T09:02:00+00:00", 1840.00),
        _case("C-2", "2026-10-21T09:11:00+00:00", 2275.50),
        current,
    ]
    state = {"case_id": "C-3", "input_data": current, "agent_logs": []}

    result = run(state, EmptyGraph(), all_cases=all_cases)

    assert result["model_results"]["fusion"]["base_risk_score"] == 0.0833
    assert result["risk_score"] == 0.35
    assert result["risk_level"] == "MEDIUM"
    assert result["model_results"]["fusion"]["risk_adjustment"]["signal"] == "high_transaction_velocity"
