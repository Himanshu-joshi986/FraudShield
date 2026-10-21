import pytest


def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["db_connected"] is True


def test_create_case(client):
    r = client.post("/api/cases", json={
        "source": "ieee_cis",
        "domain": "financial_transaction",
        "transaction_id": "T-1001",
        "amount": 250.75,
        "data_source": "public_benchmark",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["case_id"] == "FS-00001"
    assert body["status"] == "new"
    assert body["risk_level"] == "UNSCORED"

    from app.database.models import Transaction
    from app.database.session import SessionLocal
    with SessionLocal() as db:
        transaction = db.query(Transaction).filter(Transaction.transaction_id == "T-1001").one()
        assert transaction.amount == 250.75
        assert transaction.raw_features == {"entry_type": "case_intake"}


def test_create_case_persists_linked_source_record(client):
    created = client.post("/api/cases", json={
        "source": "manual",
        "domain": "insurance_claim",
        "customer_id": "C-1",
        "customer_name": "Example Customer",
        "claim_id": "CLM-MANUAL-1",
        "amount": 750.50,
        "data_source": "user_submitted",
    })
    assert created.status_code == 200
    from app.database.models import Claim
    from app.database.session import SessionLocal
    with SessionLocal() as db:
        claim = db.query(Claim).filter(Claim.claim_id == "CLM-MANUAL-1").one()
        assert claim.customer_id == "C-1"
        assert claim.amount == 750.50
        assert claim.raw_features == {"entry_type": "case_intake"}


@pytest.mark.parametrize("payload", [
    {"source": "manual", "domain": "financial_transaction", "amount": 0},
    {"source": "manual", "domain": "financial_transaction", "amount": -1},
    {"source": "manual", "domain": "financial_transaction", "amount": "NaN"},
    {"source": "manual", "domain": "insurance_claim", "transaction_id": "T-1", "amount": 1},
    {"source": "ieee_cis", "domain": "insurance_claim", "amount": 1},
])
def test_create_case_rejects_invalid_intake(client, payload):
    response = client.post("/api/cases", json=payload)
    assert response.status_code == 422


def test_list_cases_and_filter(client):
    client.post("/api/cases", json={
        "source": "ieee_cis", "domain": "financial_transaction",
        "amount": 100.0, "data_source": "public_benchmark",
    })
    client.post("/api/cases", json={
        "source": "synthetic", "domain": "insurance_claim",
        "amount": 5000.0, "data_source": "synthetic", "scenario_id": "FS-SC-003",
    })

    all_cases = client.get("/api/cases").json()
    assert len(all_cases) == 2

    synthetic_only = client.get("/api/cases?data_source=synthetic").json()
    assert len(synthetic_only) == 1
    assert synthetic_only[0]["scenario_id"] == "FS-SC-003"


def test_get_case_detail_includes_empty_sub_resources(client):
    created = client.post("/api/cases", json={
        "source": "carclaims", "domain": "insurance_claim",
        "claim_id": "CLM-1", "amount": 7500.0, "data_source": "public_benchmark",
    }).json()

    detail = client.get(f"/api/cases/{created['case_id']}").json()
    assert detail["case"]["case_id"] == created["case_id"]
    assert detail["agent_executions"] == []
    assert detail["evidence"] == []
    assert detail["reports"] == []
    assert detail["decisions"] == []


def test_get_nonexistent_case_returns_404(client):
    r = client.get("/api/cases/FS-99999")
    assert r.status_code == 404


def test_case_ids_are_sequential(client):
    ids = []
    for i in range(3):
        r = client.post("/api/cases", json={
            "source": "synthetic", "domain": "insurance_claim",
            "amount": 100.0 * (i + 1), "data_source": "synthetic",
        })
        ids.append(r.json()["case_id"])
    assert ids == ["FS-00001", "FS-00002", "FS-00003"]
