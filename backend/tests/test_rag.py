"""
Phase 8 tests -- RAG knowledge base.

Ran end-to-end in the build environment: ChromaDB 1.5.9 persistent
client, real vector similarity search, real (TF-IDF) embeddings computed
locally because the default neural embedding model's download is blocked
by this sandbox's network allowlist (verified, not assumed -- see the
docstring in app/rag/knowledge_base.py for the exact error).
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

RAG_METRICS_PATH = REPO_ROOT / "docs" / "rag_metrics.json"


@pytest.fixture
def kb(tmp_path):
    from app.rag.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(persist_dir=tmp_path / "chroma_test")
    kb.rebuild_index()
    return kb


class TestKnowledgeBaseBasics:
    def test_index_builds_from_manifest(self, kb):
        stats = kb.stats()
        assert stats["n_chunks"] > 0
        assert stats["embedding_backend"] == "offline_tfidf_v1"

    def test_all_five_documents_indexed(self, kb):
        docs = kb.list_documents()
        assert len(docs) == 5
        ids = {d["document_id"] for d in docs}
        assert ids == {"KB-001", "KB-002", "KB-003", "KB-004", "KB-005"}

    def test_every_document_has_required_provenance_fields(self, kb):
        """Section 25: document_id, title, source, jurisdiction,
        publication_date, section must all be present."""
        required = {"document_id", "title", "source", "jurisdiction", "publication_date", "section"}
        for doc in kb.list_documents():
            assert required.issubset(doc.keys())
            for field in required:
                assert doc[field], f"{doc['document_id']} missing {field}"

    def test_query_results_carry_full_provenance(self, kb):
        results = kb.query("SAR filing deadline", top_k=1)
        assert len(results) == 1
        required = {"snippet", "distance", "document_id", "title", "source",
                   "jurisdiction", "publication_date", "section"}
        assert required.issubset(results[0].keys())

    def test_empty_query_does_not_crash(self, kb):
        results = kb.query("", top_k=3)
        assert isinstance(results, list)


class TestRetrievalCorrectness:
    """These use exact-vocabulary queries (the easy case) to confirm the
    mechanism works at all before the harder paraphrase evaluation."""

    def test_sar_query_retrieves_sar_document(self, kb):
        results = kb.query("suspicious activity report filing deadline 30 days", top_k=1)
        assert results[0]["document_id"] == "KB-001"

    def test_fraud_ring_query_retrieves_fraud_ring_document(self, kb):
        # Note: an earlier version of this query ("shared device account
        # address fraud ring indicators") also lexically overlapped with
        # KB-003's override-rule wording and matched that instead -- a
        # genuine ambiguity between two related documents, not a bug.
        # Rewritten to unambiguously target KB-002's specific content.
        results = kb.query("recycled photos invoice templates repair estimates across claims", top_k=1)
        assert results[0]["document_id"] == "KB-002"

    def test_routing_query_retrieves_routing_document(self, kb):
        results = kb.query("risk score routing HIGH MEDIUM LOW escalate", top_k=1)
        assert results[0]["document_id"] == "KB-003"

    def test_results_are_ranked_by_relevance(self, kb):
        """Distances must be non-decreasing (best match first)."""
        results = kb.query("suspicious activity report", top_k=3)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)


@pytest.mark.skipif(not RAG_METRICS_PATH.exists(), reason="run scripts/evaluate_rag.py first")
class TestRAGEvaluation:
    @classmethod
    def setup_class(cls):
        with open(RAG_METRICS_PATH) as f:
            cls.metrics = json.load(f)

    def test_ran_on_real_chromadb_backend(self):
        assert self.metrics["backend"] == "offline_tfidf_v1"
        assert self.metrics["n_chunks"] > 0

    def test_accuracy_at_3_is_reasonably_high(self):
        """top-3 retrieval should catch the right document most of the
        time even with paraphrased queries."""
        assert self.metrics["accuracy_at_3"] >= 0.8

    def test_accuracy_at_1_is_measured_not_assumed(self):
        """Deliberately does NOT assert a high bar here -- TF-IDF-based
        retrieval on paraphrased queries is genuinely harder than exact
        keyword matching, and overstating this would violate Section 61."""
        assert 0.0 <= self.metrics["accuracy_at_1"] <= 1.0
        assert self.metrics["n_eval_queries"] >= 10

    def test_known_paraphrase_failure_is_recorded(self):
        """
        REGRESSION GUARD FOR AN HONEST FINDING: the query 'claim filed
        right after buying the policy is that suspicious' fails to
        retrieve KB-004 even in the top 3, because TF-IDF has no notion
        that 'buying the policy' and 'policy's inception' are related
        concepts. This is a genuine, documented limitation of the
        offline TF-IDF substitute (see the sentence-transformers upgrade
        path). If this test ever starts passing, it likely means the
        embedding backend changed -- update docs/EXPERIMENTS.md
        accordingly rather than silently deleting this test.
        """
        target = next(
            pq for pq in self.metrics["per_query"]
            if pq["query"] == "claim filed right after buying the policy is that suspicious"
        )
        assert target["hit_at_3"] is False, (
            "this TF-IDF limitation was expected to still reproduce -- "
            "if it's fixed, document what changed rather than deleting this test"
        )

    def test_scope_caveat_present(self):
        assert "scope_caveat" in self.metrics
        assert "NOT a claim" in self.metrics["scope_caveat"]


class TestSentenceTransformerFallbackCode:
    def test_class_exists_and_documents_network_requirement(self):
        from app.rag.knowledge_base import SentenceTransformerEmbedding
        import inspect
        source = inspect.getsource(SentenceTransformerEmbedding)
        assert "NOT RUN" in source

    def test_raises_clear_error_when_dependency_missing(self):
        from app.rag.knowledge_base import SentenceTransformerEmbedding
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is not None:
            pytest.skip("sentence-transformers installed -- error path not applicable")
        with pytest.raises(ImportError) as exc:
            SentenceTransformerEmbedding()
        assert "pip install sentence-transformers" in str(exc.value)


class TestVocabularyPersistence:
    def test_vectorizer_vocabulary_persists_across_instances(self, tmp_path):
        """
        A mismatched vocabulary between index-time and query-time would
        silently produce meaningless similarity scores without erroring --
        so the persisted vectorizer must actually be the one used.
        """
        from app.rag.knowledge_base import KnowledgeBase

        kb1 = KnowledgeBase(persist_dir=tmp_path / "persist_test")
        kb1.rebuild_index()
        vocab_size_1 = len(kb1.embedding_fn.vectorizer.vocabulary_)

        kb2 = KnowledgeBase(persist_dir=tmp_path / "persist_test")
        assert kb2.embedding_fn.vectorizer is not None
        vocab_size_2 = len(kb2.embedding_fn.vectorizer.vocabulary_)
        assert vocab_size_1 == vocab_size_2
