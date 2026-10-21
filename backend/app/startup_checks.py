"""
Docker startup check: ensures the kuzu graph store and ChromaDB RAG
index exist before uvicorn starts serving requests. Idempotent -- safe
to run on every container start; only does real work the first time (or
if the volume was wiped).

The scripts are executed from the repository-root layout inside the
container. Their output directories must be present in the image, while
the persistent database/index directories live on the mounted /data volume.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    graph_db_path = Path("/data/graph_db") if Path("/data").exists() else REPO_ROOT.parent / "data" / "graph_db"
    chroma_path = Path("/data/chroma_db") if Path("/data").exists() else REPO_ROOT.parent / "data" / "chroma_db"

    if not graph_db_path.exists():
        print("[startup] Graph store not found, building it now (scripts/ingest_graph.py)...")
        import subprocess
        subprocess.run([sys.executable, str(REPO_ROOT.parent / "scripts" / "ingest_graph.py")], check=True)
    else:
        print("[startup] Graph store found, skipping ingestion.")

    if not chroma_path.exists():
        print("[startup] RAG index not found, building it now (scripts/evaluate_rag.py)...")
        import subprocess
        subprocess.run([sys.executable, str(REPO_ROOT.parent / "scripts" / "evaluate_rag.py")], check=True)
    else:
        print("[startup] RAG index found, skipping build.")

    print("[startup] Checks complete.")


if __name__ == "__main__":
    main()
