import os
import sys
import tempfile
import pytest
from pathlib import Path

# ensure `app` package is importable when running pytest from repo root or backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="function")
def test_db_url(tmp_path):
    db_path = tmp_path / "test_fraudshield.db"
    return f"sqlite:///{db_path}"


@pytest.fixture(scope="function")
def client(test_db_url, monkeypatch):
    monkeypatch.setenv("FRAUDSHIELD_DB_URL", test_db_url)

    import importlib
    import sys

    def fresh_import(module_name):
        """Import a module, reloading it only if it was already cached from
        a previous test (so settings/engine bind to this test's isolated DB)."""
        if module_name in sys.modules:
            return importlib.reload(sys.modules[module_name])
        return importlib.import_module(module_name)

    fresh_import("app.core.config")
    fresh_import("app.database.session")
    fresh_import("app.database.models")
    fresh_import("app.api.cases")
    fresh_import("app.api.analytics")
    fresh_import("app.api.knowledge")
    main_module = fresh_import("app.main")

    from fastapi.testclient import TestClient
    with TestClient(main_module.app) as c:
        yield c
