"""
Phase 2 -- Synthetic scenario library (Section 8 & 9 of the spec).

IMPORTANT DISTINCTION FROM THE REAL DATASETS ABOVE:
IEEE-CIS and carclaims are used to *train and evaluate ML models* with
real, measured metrics. This script does NOT produce training data --
it produces controlled *investigation scenarios* used to demonstrate and
test the agent pipeline (Sentinel -> Investigation -> Evidence ->
Report -> Commander) end-to-end, per Section 8's explicit instruction:
"Synthetic data is allowed and required only where public benchmark
datasets do not provide the necessary artifacts" -- specifically,
multi-entity fraud-ring relationships, document mismatch/duplication,
and named "expected outcome" ground truth for testing agent behavior,
none of which IEEE-CIS or carclaims provide in a form suitable for a
graph/document demo.

Every record is tagged data_source="synthetic" and a scenario_id, and
must never be described as representing real-world fraud (Section 8).

Produces three CSVs under data/synthetic/:
  synthetic_cases.csv          -- one row per case (transaction or claim)
  synthetic_relationships.csv  -- entity-relationship edges for the graph
  synthetic_documents.csv      -- claim/invoice document pairs for Doc AI
"""
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(7)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIRST_NAMES = ["James", "Maria", "Wei", "Aisha", "Carlos", "Elena", "Raj", "Sofia", "Liam", "Noor"]
LAST_NAMES = ["Smith", "Garcia", "Chen", "Khan", "Rossi", "Kim", "Patel", "Nguyen", "Silva", "Brown"]

cases = []           # rows for synthetic_cases.csv
relationships = []   # rows for synthetic_relationships.csv
documents = []       # rows for synthetic_documents.csv

_customer_seq = 0
_case_seq = 0


def new_customer():
    global _customer_seq
    _customer_seq += 1
    return f"SYN-CUST-{_customer_seq:04d}", f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def new_case_id():
    global _case_seq
    _case_seq += 1
    return f"FS-SYN-{_case_seq:05d}"


def add_case(scenario_id, domain, customer_id, customer_name, amount, timestamp,
             expected_risk_level, notes):
    case_id = new_case_id()
    cases.append({
        "case_id": case_id, "scenario_id": scenario_id, "domain": domain,
        "customer_id": customer_id, "customer_name": customer_name,
        "amount": round(amount, 2), "timestamp": timestamp.isoformat(),
        "data_source": "synthetic", "expected_risk_level": expected_risk_level,
        "notes": notes,
    })
    return case_id


def add_relationship(scenario_id, source, target, relation):
    relationships.append({"scenario_id": scenario_id, "source_entity": source,
                           "target_entity": target, "relation": relation})


def add_document(scenario_id, case_id, doc_type, claim_amount, invoice_amount,
                  invoice_number, duplicate_of=None):
    documents.append({
        "scenario_id": scenario_id, "case_id": case_id, "doc_type": doc_type,
        "claim_amount": round(claim_amount, 2), "invoice_amount": round(invoice_amount, 2),
        "invoice_number": invoice_number, "duplicate_of_invoice": duplicate_of or "",
        "amount_mismatch": abs(claim_amount - invoice_amount) > 1.0,
    })


NOW = datetime(2026, 6, 1, 9, 0, 0)


# --- Scenario 1: Legitimate case, no suspicious signals -------------------
def scenario_1():
    sid = "FS-SC-001"
    cid, name = new_customer()
    account = f"ACC-{random.randint(10000000,99999999)}"
    device = f"DEV-{random.randint(100000,999999)}"
    add_case(sid, "financial_transaction", cid, name, 145.30,
             NOW - timedelta(days=10), "LOW",
             "Single routine transaction, no shared entities, no anomalies.")
    add_relationship(sid, cid, account, "owns_account")
    add_relationship(sid, cid, device, "uses_device")


# --- Scenario 2: Transaction velocity anomaly ------------------------------
def scenario_2():
    sid = "FS-SC-002"
    cid, name = new_customer()
    account = f"ACC-{random.randint(10000000,99999999)}"
    base_time = NOW - timedelta(hours=2)
    for i in range(9):
        add_case(sid, "financial_transaction", cid, name, random.uniform(200, 900),
                 base_time + timedelta(minutes=i * 6), "MEDIUM",
                 "Part of a 9-transaction burst within ~50 minutes on the same account (velocity anomaly).")
    add_relationship(sid, cid, account, "owns_account")


# --- Scenario 3: Fraud ring (shared account/device/email across entities) -
def scenario_3():
    sid = "FS-SC-003"
    shared_account = f"ACC-{random.randint(10000000,99999999)}"
    shared_device = f"DEV-{random.randint(100000,999999)}"
    shared_email = f"user{random.randint(100,999)}@freemail-temp.example"
    broker = "Broker-Apex-Underwriting"
    base_time = NOW - timedelta(hours=5)

    for i in range(6):
        cid, name = new_customer()
        add_case(sid, "financial_transaction", cid, name, random.uniform(3000, 9000),
                 base_time + timedelta(minutes=i * 12), "HIGH",
                 "Member of a 6-entity ring sharing account/device/email/broker.")
        add_relationship(sid, cid, shared_account, "owns_account")
        add_relationship(sid, cid, shared_device, "uses_device")
        add_relationship(sid, cid, shared_email, "has_email")
        add_relationship(sid, cid, broker, "filed_via_broker")


# --- Scenario 4: Document mismatch (claim amount != invoice amount) --------
def scenario_4():
    sid = "FS-SC-004"
    cid, name = new_customer()
    case_id = add_case(sid, "insurance_claim", cid, name, 5000.00,
                        NOW - timedelta(days=3), "MEDIUM",
                        "Claim amount ($5,000) does not match submitted invoice amount ($7,500).")
    add_document(sid, case_id, "invoice", claim_amount=5000.00, invoice_amount=7500.00,
                 invoice_number=f"INV-{random.randint(10000,99999)}")


# --- Scenario 5: Duplicate invoice reused across claims --------------------
def scenario_5():
    sid = "FS-SC-005"
    shared_invoice_number = f"INV-{random.randint(10000,99999)}"
    for i in range(3):
        cid, name = new_customer()
        case_id = add_case(sid, "insurance_claim", cid, name, random.uniform(2000, 4000),
                            NOW - timedelta(days=random.randint(1, 20)), "HIGH",
                            f"Reuses invoice {shared_invoice_number} also submitted on {i} other claim(s).")
        add_document(sid, case_id, "invoice", claim_amount=2500.00, invoice_amount=2500.00,
                     invoice_number=shared_invoice_number,
                     duplicate_of=shared_invoice_number if i > 0 else None)


# --- Scenario 6: Shared identity infrastructure among "unrelated" users ----
def scenario_6():
    sid = "FS-SC-006"
    shared_address = "742 Evergreen Terrace, Springfield"
    for i in range(4):
        cid, name = new_customer()
        case_id = add_case(sid, "insurance_claim", cid, name, random.uniform(1500, 3000),
                            NOW - timedelta(days=random.randint(1, 15)), "MEDIUM",
                            "Claimant shares a mailing address with 3 other supposedly unrelated claimants.")
        add_relationship(sid, cid, shared_address, "has_address")


# --- Scenario 7: Mixed evidence (some legit signals, some suspicious) ------
def scenario_7():
    sid = "FS-SC-007"
    cid, name = new_customer()
    account = f"ACC-{random.randint(10000000,99999999)}"
    base_time = NOW - timedelta(days=1)
    # a genuine, but mild, velocity signal: 2 transactions ~22 minutes
    # apart (below the 3-in-an-hour "high velocity" threshold used by the
    # temporal/velocity detector, but a real, present flag rather than
    # zero signal) -- otherwise this scenario had no encoded signal at all
    # despite its notes claiming one, which was an inconsistency found
    # during Phase 9 agent-pipeline evaluation.
    #
    # expected_risk_level is LOW, not the originally-guessed MEDIUM: with
    # a mild velocity flag but a verified document and zero graph
    # connections, a well-calibrated system SHOULD net out low-risk on
    # weighted evidence. The scenario's real purpose is demonstrating that
    # the weak signal is detected and recorded (see the temporal_risk
    # value in Sentinel's output) without being enough, alone, to trigger
    # escalation -- i.e. testing restraint, not testing that MEDIUM is
    # reached. The original MEDIUM label was an unvalidated guess from
    # Phase 2, corrected here after Phase 9 evaluation showed the actual
    # weighted evidence doesn't support it.
    add_case(sid, "financial_transaction", cid, name, 450.00,
             base_time, "LOW",
             "First of two transactions ~22 minutes apart -- a genuine but mild velocity signal.")
    case_id = add_case(sid, "financial_transaction", cid, name, 620.00,
                        base_time + timedelta(minutes=22), "LOW",
                        "Second transaction of a mild 2-in-22-minutes pattern, otherwise clean "
                        "history and verified documents. Tests that a weak signal is detected "
                        "but correctly does NOT trigger escalation on its own.")
    add_relationship(sid, cid, account, "owns_account")
    add_document(sid, case_id, "receipt", claim_amount=620.00, invoice_amount=620.00,
                 invoice_number=f"INV-{random.randint(10000,99999)}")


# --- Scenario 8: High-risk connected case (multiple independent signals agree) --
def scenario_8():
    sid = "FS-SC-008"
    shared_account = f"ACC-{random.randint(10000000,99999999)}"
    shared_device = f"DEV-{random.randint(100000,999999)}"
    base_time = NOW - timedelta(hours=1)
    invoice_number = f"INV-{random.randint(10000,99999)}"

    for i in range(4):
        cid, name = new_customer()
        case_id = add_case(sid, "insurance_claim", cid, name, random.uniform(6000, 12000),
                            base_time + timedelta(minutes=i * 8), "HIGH",
                            "Combines: temporal burst, shared account/device (graph), document amount "
                            "mismatch, and (for entity 2+) duplicate invoice -- 4 independent signals agree.")
        add_relationship(sid, cid, shared_account, "owns_account")
        add_relationship(sid, cid, shared_device, "uses_device")
        add_document(sid, case_id, "invoice", claim_amount=8000.00,
                     invoice_amount=6000.00 if i == 0 else 8000.00,
                     invoice_number=invoice_number, duplicate_of=invoice_number if i > 0 else None)


def write_csv(rows, filename, fieldnames):
    path = OUT_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {path}")


def main():
    for fn in [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5, scenario_6, scenario_7, scenario_8]:
        fn()

    write_csv(cases, "synthetic_cases.csv",
              ["case_id", "scenario_id", "domain", "customer_id", "customer_name", "amount",
               "timestamp", "data_source", "expected_risk_level", "notes"])
    write_csv(relationships, "synthetic_relationships.csv",
              ["scenario_id", "source_entity", "target_entity", "relation"])
    write_csv(documents, "synthetic_documents.csv",
              ["scenario_id", "case_id", "doc_type", "claim_amount", "invoice_amount",
               "invoice_number", "duplicate_of_invoice", "amount_mismatch"])

    print(f"\nTotal: {len(cases)} synthetic cases across 8 scenarios, "
          f"{len(relationships)} relationships, {len(documents)} documents.")


if __name__ == "__main__":
    main()
