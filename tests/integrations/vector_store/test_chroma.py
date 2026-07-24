import pytest

pytest.importorskip("langchain_chroma")

from agent_platform.integrations.credentials import ChromaCredentials
from agent_platform.integrations.vector_store.chroma.chroma import ChromaStore
from agent_platform.integrations.vector_store.chroma.config import ChromaConfig


def test_chroma_config_persist_directory_defaults_to_none():
    assert ChromaConfig().persist_directory is None


def test_chroma_config_ssl_defaults_false():
    assert ChromaConfig().ssl is False


def test_chroma_config_tenant_database_defaults():
    cfg = ChromaConfig()
    assert cfg.tenant == "default_tenant"
    assert cfg.database == "default_database"


def test_chroma_config_custom_fields():
    cfg = ChromaConfig(
        persist_directory="/tmp/chroma",
        ssl=True,
        tenant="acme",
        database="prod",
    )
    assert cfg.persist_directory == "/tmp/chroma"
    assert cfg.ssl is True
    assert cfg.tenant == "acme"
    assert cfg.database == "prod"


class TestChromaBuildClient:
    def _patch(self, monkeypatch):
        import agent_platform.integrations.vector_store.chroma.chroma as mod

        captured = {}

        class FakeChroma:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(mod, "Chroma", FakeChroma)
        return captured

    def test_persist_directory_uses_embedded_mode(self, monkeypatch):
        captured = self._patch(monkeypatch)

        store = ChromaStore.__new__(ChromaStore)
        store._credentials = ChromaCredentials(api_key=None)
        store._embeddings = None

        store._build_client(ChromaConfig(persist_directory="/tmp/chroma"))

        assert captured["persist_directory"] == "/tmp/chroma"
        assert "host" not in captured
        assert "port" not in captured

    def test_no_persist_directory_uses_http_client_mode(self, monkeypatch):
        captured = self._patch(monkeypatch)

        store = ChromaStore.__new__(ChromaStore)
        store._credentials = ChromaCredentials(api_key=None)
        store._embeddings = None

        store._build_client(ChromaConfig(host="10.0.0.5", port=9000, ssl=True))

        assert captured["host"] == "10.0.0.5"
        assert captured["port"] == 9000
        assert captured["ssl"] is True
        assert "persist_directory" not in captured

    def test_tenant_and_database_always_forwarded(self, monkeypatch):
        captured = self._patch(monkeypatch)

        store = ChromaStore.__new__(ChromaStore)
        store._credentials = ChromaCredentials(api_key=None)
        store._embeddings = None

        store._build_client(ChromaConfig(tenant="acme", database="prod"))

        assert captured["tenant"] == "acme"
        assert captured["database"] == "prod"

    def test_api_key_forwarded_when_present(self, monkeypatch):
        captured = self._patch(monkeypatch)

        store = ChromaStore.__new__(ChromaStore)
        store._credentials = ChromaCredentials(api_key="cloud-key")
        store._embeddings = None

        store._build_client(ChromaConfig())

        assert captured["chroma_cloud_api_key"] == "cloud-key"

    def test_api_key_omitted_when_absent(self, monkeypatch):
        captured = self._patch(monkeypatch)

        store = ChromaStore.__new__(ChromaStore)
        store._credentials = ChromaCredentials(api_key=None)
        store._embeddings = None

        store._build_client(ChromaConfig())

        assert "chroma_cloud_api_key" not in captured
