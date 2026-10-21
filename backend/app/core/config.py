"""
Centralized configuration. Every external dependency (DB, graph store,
vector store, LLM provider) is read from environment variables so the
system can move from local-first (SQLite/kuzu/Chroma) to production
(Postgres/Neo4j/hosted Chroma) infrastructure with zero code changes --
only .env changes. See .env.example for the full documented list.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # repo root


class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"

    fraudshield_db_url: str = f"sqlite:///{BASE_DIR / 'backend' / 'fraudshield.db'}"

    graph_backend: str = "kuzu"
    kuzu_db_path: str = str(BASE_DIR / "data" / "graph_db")
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"

    chroma_persist_dir: str = str(BASE_DIR / "data" / "chroma_db")

    llm_provider: str = "gemini"
    llm_model: str = "gemini-1.5-flash"
    gemini_api_key: str = ""
    llm_max_requests_per_day: int = 400
    llm_max_requests_per_minute: int = 10

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")


settings = Settings()
