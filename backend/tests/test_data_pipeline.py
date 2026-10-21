"""
Phase 2 tests -- these check the *actual* processed files on disk (not
mocks), so they double as a regression check: if someone changes the
preprocessing scripts and breaks the split logic or introduces leakage,
these tests catch it.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"
SYNTHETIC = REPO_ROOT / "data" / "synthetic"


@pytest.mark.skipif(not (PROCESSED / "ieee_cis_train.parquet").exists(), reason="run scripts/preprocess_ieee.py first")
class TestIEEECISPipeline:
    @classmethod
    def setup_class(cls):
        cls.train = pd.read_parquet(PROCESSED / "ieee_cis_train.parquet")
        cls.val = pd.read_parquet(PROCESSED / "ieee_cis_val.parquet")
        cls.test = pd.read_parquet(PROCESSED / "ieee_cis_test.parquet")
        with open(PROCESSED / "ieee_cis_stats.json") as f:
            cls.stats = json.load(f)

    def test_row_counts_match_raw_total(self):
        assert len(self.train) + len(self.val) + len(self.test) == self.stats["total_rows"]
        assert self.stats["total_rows"] == 590540  # actual IEEE-CIS train_transaction.csv row count

    def test_no_temporal_leakage_between_splits(self):
        """The whole point of a time-aware split: train must end before val starts,
        val must end before test starts. A random split would fail this."""
        assert self.train["TransactionDT"].max() <= self.val["TransactionDT"].min()
        assert self.val["TransactionDT"].max() <= self.test["TransactionDT"].min()

    def test_fraud_label_is_binary_and_present(self):
        for split in [self.train, self.val, self.test]:
            assert set(split["isFraud"].unique()).issubset({0, 1})
            assert split["isFraud"].sum() > 0  # each split has at least some fraud examples

    def test_fraud_rate_is_realistic_not_fabricated(self):
        # the real IEEE-CIS fraud rate is ~3.5% -- if this were fabricated
        # synthetic data it would likely be a round number like 5% or 10%
        overall_rate = (
            self.train["isFraud"].sum() + self.val["isFraud"].sum() + self.test["isFraud"].sum()
        ) / self.stats["total_rows"]
        assert 0.03 < overall_rate < 0.04

    def test_engineered_features_present(self):
        for col in ["txn_hour", "txn_dayofweek", "TransactionAmt_log", "card1_freq", "has_identity"]:
            assert col in self.train.columns

    def test_identity_coverage_matches_known_value(self):
        # 144,233 identity rows / 590,540 transaction rows = 24.42%
        assert abs(self.stats["identity_coverage_pct"] - 24.42) < 0.1


@pytest.mark.skipif(not (PROCESSED / "carclaims_train.parquet").exists(), reason="run scripts/preprocess_carclaims.py first")
class TestCarclaimsPipeline:
    @classmethod
    def setup_class(cls):
        cls.train = pd.read_parquet(PROCESSED / "carclaims_train.parquet")
        cls.val = pd.read_parquet(PROCESSED / "carclaims_val.parquet")
        cls.test = pd.read_parquet(PROCESSED / "carclaims_test.parquet")

    def test_row_counts_match_raw_total(self):
        assert len(self.train) + len(self.val) + len(self.test) == 15420

    def test_stratified_split_preserves_fraud_rate(self):
        rates = [split["fraud_label"].mean() for split in [self.train, self.val, self.test]]
        # all three splits' fraud rates should be within 0.5 percentage points of each other
        assert max(rates) - min(rates) < 0.005

    def test_claim_ids_are_unique_across_splits(self):
        all_ids = pd.concat([self.train["claim_id"], self.val["claim_id"], self.test["claim_id"]])
        assert all_ids.is_unique


@pytest.mark.skipif(not (SYNTHETIC / "synthetic_cases.csv").exists(), reason="run scripts/generate_synthetic_scenarios.py first")
class TestSyntheticScenarios:
    @classmethod
    def setup_class(cls):
        cls.cases = pd.read_csv(SYNTHETIC / "synthetic_cases.csv")
        cls.relationships = pd.read_csv(SYNTHETIC / "synthetic_relationships.csv")
        cls.documents = pd.read_csv(SYNTHETIC / "synthetic_documents.csv")

    def test_all_eight_scenarios_present(self):
        expected = {f"FS-SC-{i:03d}" for i in range(1, 9)}
        assert set(self.cases["scenario_id"].unique()) == expected

    def test_all_records_tagged_as_synthetic(self):
        assert (self.cases["data_source"] == "synthetic").all()

    def test_case_ids_are_unique(self):
        assert self.cases["case_id"].is_unique

    def test_scenario_3_is_a_real_fraud_ring(self):
        """Scenario 3 should have 6 distinct customers all sharing at least
        one account/device/email/broker entity -- verifying the graph
        structure, not just that rows exist."""
        ring_relationships = self.relationships[self.relationships["scenario_id"] == "FS-SC-003"]
        shared_targets = ring_relationships.groupby("target_entity")["source_entity"].nunique()
        assert (shared_targets >= 6).any(), "expected at least one entity shared by all 6 ring members"

    def test_scenario_4_document_mismatch_is_real(self):
        doc = self.documents[self.documents["scenario_id"] == "FS-SC-004"].iloc[0]
        assert doc["amount_mismatch"] == True
        assert doc["claim_amount"] != doc["invoice_amount"]

    def test_scenario_5_has_duplicate_invoice(self):
        docs = self.documents[self.documents["scenario_id"] == "FS-SC-005"]
        assert docs["invoice_number"].duplicated().any()

    def test_expected_risk_levels_are_valid(self):
        assert set(self.cases["expected_risk_level"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})
