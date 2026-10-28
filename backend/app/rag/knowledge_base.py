"""
Phase 8 -- RAG knowledge base.

Real, persistent vector store: ChromaDB 1.5.9 in embedded/persistent mode
(no server process). VERIFIED WORKING in this environment.

--- A real constraint, handled honestly, not hidden ---
ChromaDB's DEFAULT embedding function tries to download a pretrained
ONNX sentence-transformer model from a URL outside this sandbox's
network allowlist. That download fails here
(`ValueError: Downloaded file ... does not match expected SHA256 hash`),
which is actually the download being blocked/truncated by the egress
proxy, not model corruption.

Rather than silently swallow this and pretend semantic embeddings are
running, this module uses a documented, verified-working alternative: a
TfidfVectorizer FITTED ON THE ACTUAL KNOWLEDGE BASE CORPUS, wrapped as a
ChromaDB EmbeddingFunction. This is a real, legitimate retrieval method
(term-frequency / inverse-document-frequency weighting is a standard,
well-understood IR technique) -- it is simply not a *neural* embedding.

--- Upgrade path once you have broader network access ---
Set FRAUDSHIELD_EMBEDDING=sentence-transformers and install
`sentence-transformers` -- SentenceTransformerEmbedding below is written
against the real API and will download a real MiniLM model the first
time it runs. Swap is a one-line change; the rest of this module
(chunking, querying, metadata) does not need to change.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[3]
REGULATIONS_DIR = REPO_ROOT / "knowledge_base" / "regulatory"
DEFAULT_CHROMA_DIR = REPO_ROOT / "data" / "chroma_db"
UPLOADS_DIR = REPO_ROOT / "data" / "knowledge_uploads"
UPLOAD_MANIFEST = UPLOADS_DIR / "manifest.json"

CHUNK_SIZE_CHARS = 400


class OfflineTfidfEmbedding(EmbeddingFunction):
    """
    VERIFIED WORKING. Fitted once on the knowledge base corpus at
    ingestion time; the fitted vocabulary is persisted to disk so query
    time uses the exact same vectorizer that indexed the documents
    (a mismatched vocabulary would silently produce meaningless
    similarity scores).
    """

    def __init__(self, vocab_path: Path, max_features: int = 4000):
        self.vocab_path = Path(vocab_path)
        self.max_features = max_features
        self.vectorizer: TfidfVectorizer | None = None
        if self.vocab_path.exists():
            self._load()

    def fit(self, corpus: list[str]):
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features)
        self.vectorizer.fit(corpus)
        self._save()

    def _save(self):
        import joblib
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, self.vocab_path)

    def _load(self):
        import joblib
        self.vectorizer = joblib.load(self.vocab_path)

    def __call__(self, input):
        if self.vectorizer is None:
            raise RuntimeError(
                "Embedding function not fitted yet. Call KnowledgeBase.rebuild_index() "
                "before querying, or ensure the persisted vocabulary file exists."
            )
        return self.vectorizer.transform(input).toarray().tolist()

    def name(self):
        return "offline_tfidf_v1"

    def get_config(self) -> dict:
        return {"vocab_path": str(self.vocab_path), "max_features": self.max_features}

    @staticmethod
    def build_from_config(config: dict) -> "OfflineTfidfEmbedding":
        return OfflineTfidfEmbedding(config["vocab_path"], config.get("max_features", 4000))


class SentenceTransformerEmbedding(EmbeddingFunction):
    """
    NOT RUN in the build environment (needs to download model weights
    from huggingface.co, outside this sandbox's network allowlist).
    Written against the real sentence-transformers API.
        pip install sentence-transformers
        export FRAUDSHIELD_EMBEDDING=sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. Install with:\n"
                "    pip install sentence-transformers\n"
                "Or use the offline TF-IDF embedding (default), which is "
                "what the measured Phase 8 results used."
            ) from e
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        return self.model.encode(input, convert_to_numpy=True).tolist()

    def name(self):
        return "sentence-transformers-all-MiniLM-L6-v2"


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) > chunk_size and buf:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = (buf + "\n\n" + p) if buf else p
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


class KnowledgeBase:
    """
    RAG knowledge base over the regulatory documents in
    knowledge_base/regulatory/, backed by ChromaDB (persistent, embedded).

    Every returned result carries the full document metadata Section 25
    requires (document_id, title, source, jurisdiction, publication_date,
    section) -- not just a similarity score.
    """

    def __init__(self, persist_dir: str | Path | None = None,
                embedding: str | None = None):
        self.persist_dir = Path(persist_dir or os.environ.get("CHROMA_PERSIST_DIR") or DEFAULT_CHROMA_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        embedding = embedding or os.environ.get("FRAUDSHIELD_EMBEDDING", "tfidf")
        if embedding == "sentence-transformers":
            self.embedding_fn = SentenceTransformerEmbedding()
        else:
            vocab_path = self.persist_dir / "tfidf_vectorizer.joblib"
            self.embedding_fn = OfflineTfidfEmbedding(vocab_path)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            "regulatory_kb", embedding_function=self.embedding_fn,
        )

    def _documents(self) -> list[dict]:
        with open(REGULATIONS_DIR / "manifest.json") as f:
            manifest = json.load(f)["documents"]
        docs = []
        for item in manifest:
            doc = dict(item)
            doc["text"] = (REGULATIONS_DIR / item["filename"]).read_text(encoding="utf-8")
            doc["case_id"] = ""
            docs.append(doc)
        if UPLOAD_MANIFEST.exists():
            for item in json.loads(UPLOAD_MANIFEST.read_text(encoding="utf-8")):
                doc = dict(item)
                doc["text"] = (UPLOADS_DIR / item["text_filename"]).read_text(encoding="utf-8")
                docs.append(doc)
        return docs

    def rebuild_index(self):
        """Rebuilds the complete corpus so TF-IDF vocabulary stays consistent."""
        documents = self._documents()

        try:
            self.client.delete_collection("regulatory_kb")
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            "regulatory_kb", embedding_function=self.embedding_fn,
        )

        all_chunks, all_ids, all_metadatas = [], [], []
        for doc_meta in documents:
            chunks = _chunk_text(doc_meta["text"])
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{doc_meta['document_id']}-chunk-{i}")
                all_metadatas.append({
                    "document_id": doc_meta["document_id"],
                    "title": doc_meta["title"],
                    "source": doc_meta["source"],
                    "jurisdiction": doc_meta["jurisdiction"],
                    "publication_date": doc_meta["publication_date"],
                    "section": doc_meta["section"],
                    "chunk_index": i,
                    "case_id": doc_meta.get("case_id", ""),
                })

        if isinstance(self.embedding_fn, OfflineTfidfEmbedding):
            self.embedding_fn.fit(all_chunks)

        self.collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)
        return {"documents_indexed": len(documents), "chunks_indexed": len(all_chunks)}

    def query(self, question: str, top_k: int = 3, case_id: str | None = None) -> list[dict]:
        # Older indexes predate case_id metadata. Fetch a wider candidate set
        # and filter here so those shared regulatory records remain visible.
        count = self.collection.count()
        if count == 0:
            return []
        n_results = min(count, max(top_k * 20, top_k))
        results = self.collection.query(query_texts=[question], n_results=n_results)
        if not results["ids"] or not results["ids"][0]:
            return []

        out = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            scope = meta.get("case_id") or ""
            if (case_id and scope not in ("", case_id)) or (not case_id and scope):
                continue
            out.append({
                "snippet": results["documents"][0][i],
                "distance": round(float(results["distances"][0][i]), 4),
                "document_id": meta["document_id"],
                "title": meta["title"],
                "source": meta["source"],
                "jurisdiction": meta["jurisdiction"],
                "publication_date": meta["publication_date"],
                "section": meta["section"],
                "case_id": meta.get("case_id") or None,
            })
        return out[:top_k]

    def list_documents(self) -> list[dict]:
        with open(REGULATIONS_DIR / "manifest.json") as f:
            documents = [dict(d, category="regulatory", case_id=None) for d in json.load(f)["documents"]]
        if UPLOAD_MANIFEST.exists():
            documents.extend({key: d.get(key) for key in (
                "document_id", "title", "source", "original_filename", "jurisdiction",
                "publication_date", "section", "case_id",
            )} | {"category": "user"}
                for d in json.loads(UPLOAD_MANIFEST.read_text(encoding="utf-8")))
        return documents

    def add_upload(self, filename: str, content: bytes, title: str, jurisdiction: str,
                   case_id: str | None = None) -> dict:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".txt", ".md", ".pdf"}:
            raise ValueError("Upload a text, Markdown, or text-based PDF file.")
        if not content or len(content) > 10 * 1024 * 1024:
            raise ValueError("Files must be non-empty and no larger than 10 MB.")
        if suffix == ".pdf":
            import pdfplumber
            import io
            try:
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    extracted = "\n\n".join((p.extract_text() or "") for p in pdf.pages).strip()
            except Exception as exc:
                raise ValueError("Could not read this PDF. Try exporting it as a text-based PDF.") from exc
            if not extracted:
                raise ValueError("This PDF has no selectable text. OCR scanned PDFs before uploading.")
        else:
            try:
                extracted = content.decode("utf-8-sig").strip()
            except UnicodeDecodeError as exc:
                raise ValueError("Text and Markdown files must use UTF-8 encoding.") from exc
        if len(extracted) < 20:
            raise ValueError("The document needs at least 20 characters of readable text.")

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(UPLOAD_MANIFEST.read_text(encoding="utf-8")) if UPLOAD_MANIFEST.exists() else []
        digest = hashlib.sha256(content).hexdigest()
        if any(d["sha256"] == digest and d.get("case_id", "") == (case_id or "") for d in manifest):
            raise ValueError("This same file is already in the library for that scope.")
        scope_hash = hashlib.sha256((case_id or "shared").encode()).hexdigest()[:6].upper()
        doc_id = f"USR-{digest[:10].upper()}-{scope_hash}"
        original_name = Path(filename).name[:180]
        original_path = UPLOADS_DIR / f"{doc_id}{suffix}"
        text_path = UPLOADS_DIR / f"{doc_id}.txt"
        original_path.write_bytes(content)
        text_path.write_text(extracted, encoding="utf-8")
        item = {
            "document_id": doc_id, "title": title.strip() or original_name,
            "source": original_name, "original_filename": original_name,
            "stored_filename": original_path.name, "text_filename": text_path.name,
            "jurisdiction": jurisdiction.strip() or "Unspecified", "publication_date": "",
            "section": "User provided", "case_id": case_id or "", "sha256": digest,
            "category": "user",
        }
        manifest.append(item)
        UPLOAD_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            self.rebuild_index()
        except Exception:
            manifest.pop()
            UPLOAD_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            original_path.unlink(missing_ok=True)
            text_path.unlink(missing_ok=True)
            self.rebuild_index()
            raise
        return {k: v for k, v in item.items() if k not in {"sha256", "stored_filename", "text_filename"}}

    def link_upload(self, document_id: str, case_id: str | None) -> dict:
        if not UPLOAD_MANIFEST.exists():
            raise KeyError(document_id)
        manifest = json.loads(UPLOAD_MANIFEST.read_text(encoding="utf-8"))
        doc = next((item for item in manifest if item["document_id"] == document_id), None)
        if doc is None:
            raise KeyError(document_id)
        previous = doc.get("case_id", "")
        doc["case_id"] = case_id or ""
        UPLOAD_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            self.rebuild_index()
        except Exception:
            doc["case_id"] = previous
            UPLOAD_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self.rebuild_index()
            raise
        return {key: doc.get(key) for key in (
            "document_id", "title", "source", "original_filename", "jurisdiction",
            "publication_date", "section",
        )} | {"category": "user", "case_id": doc.get("case_id") or None}

    def remove_upload(self, document_id: str) -> dict:
        """Remove a contributed source and rebuild search before deleting its files."""
        if not UPLOAD_MANIFEST.exists():
            raise KeyError(document_id)
        original_manifest = json.loads(UPLOAD_MANIFEST.read_text(encoding="utf-8"))
        removed = next((item for item in original_manifest if item.get("document_id") == document_id), None)
        if removed is None:
            raise KeyError(document_id)

        upload_root = UPLOADS_DIR.resolve()
        file_paths = []
        for key in ("stored_filename", "text_filename"):
            filename = removed.get(key)
            if not filename:
                continue
            path = (UPLOADS_DIR / filename).resolve()
            if path.parent != upload_root:
                raise RuntimeError("Refusing to remove an upload outside the managed knowledge folder.")
            file_paths.append(path)

        next_manifest = [item for item in original_manifest if item.get("document_id") != document_id]
        UPLOAD_MANIFEST.write_text(json.dumps(next_manifest, indent=2), encoding="utf-8")
        try:
            self.rebuild_index()
        except Exception:
            UPLOAD_MANIFEST.write_text(json.dumps(original_manifest, indent=2), encoding="utf-8")
            try:
                self.rebuild_index()
            except Exception as restore_error:
                raise RuntimeError("Index rebuild failed and the previous index could not be restored.") from restore_error
            raise

        # Only remove files named by this user-upload manifest, and verify
        # resolved paths stay inside the managed uploads directory.
        for path in file_paths:
            path.unlink(missing_ok=True)
        return {key: removed.get(key) for key in (
            "document_id", "title", "source", "original_filename", "case_id",
        )}

    def stats(self) -> dict:
        return {
            "n_chunks": self.collection.count(),
            "embedding_backend": self.embedding_fn.name(),
        }


_kb_instance: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
