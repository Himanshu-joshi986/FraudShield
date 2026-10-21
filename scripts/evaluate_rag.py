"""
Phase 8 -- RAG index build + retrieval evaluation.

Evaluates retrieval on a hand-written query set where each query has a
KNOWN correct document (the query was written to be about that
document's specific content). This is a genuine evaluation of whether
retrieval works, not a demo of only-easy examples -- includes queries
using different vocabulary than the source document (paraphrases), which
is the harder, more realistic case for TF-IDF-based retrieval.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.rag.knowledge_base import get_knowledge_base  # noqa: E402

OUT_PATH = REPO_ROOT / "docs" / "rag_metrics.json"

# (query, expected_document_id) -- some paraphrased away from the source
# document's exact wording, to test real retrieval rather than keyword
# matching against text the query happens to quote.
EVAL_QUERIES = [
    ("how many days do we have to file a suspicious activity report", "KB-001"),
    ("what is the dollar threshold that triggers a SAR filing", "KB-001"),
    ("is splitting a claim into smaller pieces to avoid reporting a red flag", "KB-001"),
    ("what patterns suggest several claimants are working together in a ring", "KB-002"),
    ("multiple people using the same device or account to file claims", "KB-002"),
    ("what should happen if a case has a risk score of 0.8", "KB-003"),
    ("when does a fraud ring override the normal risk score routing", "KB-003"),
    ("warning signs of a fraudulent car accident claim", "KB-004"),
    ("claim filed right after buying the policy is that suspicious", "KB-004"),
    ("should the system ever say a person definitely committed fraud", "KB-005"),
    ("can someone request a human to review an automated fraud decision", "KB-005"),
]


def evaluate():
    kb = get_knowledge_base()
    build_stats = kb.rebuild_index()
    print(f"Index built: {build_stats}")

    correct_at_1 = 0
    correct_at_3 = 0
    per_query = []

    for query, expected_doc in EVAL_QUERIES:
        results = kb.query(query, top_k=3)
        top_docs = [r["document_id"] for r in results]

        hit_at_1 = len(top_docs) > 0 and top_docs[0] == expected_doc
        hit_at_3 = expected_doc in top_docs

        correct_at_1 += int(hit_at_1)
        correct_at_3 += int(hit_at_3)

        per_query.append({
            "query": query,
            "expected_document": expected_doc,
            "retrieved_documents": top_docs,
            "top_result_distance": results[0]["distance"] if results else None,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
        })

    n = len(EVAL_QUERIES)
    metrics = {
        "backend": kb.stats()["embedding_backend"],
        "n_documents": build_stats["documents_indexed"],
        "n_chunks": build_stats["chunks_indexed"],
        "n_eval_queries": n,
        "accuracy_at_1": round(correct_at_1 / n, 4),
        "accuracy_at_3": round(correct_at_3 / n, 4),
        "per_query": per_query,
        "scope_caveat": (
            "Evaluated on 11 hand-written queries against 5 knowledge-base documents. "
            "Small evaluation set for a small, purpose-built corpus. Demonstrates the "
            "retrieval mechanism works, including on paraphrased queries that don't "
            "reuse the source document's exact wording. NOT a claim about retrieval "
            "accuracy on a large, heterogeneous real regulatory corpus, where TF-IDF's "
            "lack of true semantic understanding (e.g. synonyms, paraphrase at scale) "
            "would be a much bigger limitation -- see the sentence-transformers upgrade "
            "path in knowledge_base.py for that case."
        ),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\naccuracy@1: {metrics['accuracy_at_1']}, accuracy@3: {metrics['accuracy_at_3']}")
    for pq in per_query:
        mark = "✓" if pq["hit_at_1"] else ("~" if pq["hit_at_3"] else "✗")
        print(f"  {mark} [{pq['expected_document']}] {pq['query'][:60]}")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    evaluate()
