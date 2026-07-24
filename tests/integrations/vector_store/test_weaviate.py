from unittest.mock import AsyncMock

import pytest

pytest.importorskip("weaviate")
pytest.importorskip("langchain_weaviate")

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.vector_store.weaviate.config import WeaviateConfig
from agent_platform.integrations.vector_store.weaviate.weaviate import (
    WeaviateStore,
    _parse_url,
)


@pytest.fixture
def provider():
    return WeaviateStore.__new__(WeaviateStore)


class TestWeaviateSearchKwargs:
    def test_no_filter_no_namespace_returns_empty(self, provider):
        kwargs = provider._search_kwargs(WeaviateConfig(), None)
        assert kwargs == {}

    def test_filter_uses_filters_kwarg_not_filter(self, provider):
        kwargs = provider._search_kwargs(WeaviateConfig(), {"source": "doc.txt"})
        assert "filters" in kwargs
        assert "filter" not in kwargs

    def test_single_condition_filter(self, provider):
        from weaviate.collections.classes.filters import _FilterValue

        kwargs = provider._search_kwargs(WeaviateConfig(), {"source": "doc.txt"})
        assert isinstance(kwargs["filters"], _FilterValue)

    def test_multi_key_filter_combines_with_all_of(self, provider):
        from weaviate.collections.classes.filters import _Filters

        kwargs = provider._search_kwargs(
            WeaviateConfig(), {"source": "doc.txt", "language": "en"}
        )
        assert isinstance(kwargs["filters"], _Filters)

    def test_namespace_maps_to_tenant_kwarg(self, provider):
        kwargs = provider._search_kwargs(WeaviateConfig(namespace="tenant-a"), None)
        assert kwargs == {"tenant": "tenant-a"}

    def test_no_namespace_omits_tenant(self, provider):
        kwargs = provider._search_kwargs(WeaviateConfig(), None)
        assert "tenant" not in kwargs

    def test_build_client_enables_multi_tenancy_when_namespace_set(
        self, provider, monkeypatch
    ):
        import agent_platform.integrations.vector_store.weaviate.weaviate as mod

        captured = {}

        class FakeWeaviateVectorStore:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(mod, "WeaviateVectorStore", FakeWeaviateVectorStore)
        provider._credentials = mod.WeaviateCredentials(api_key=None)

        class FakeClient:
            pass

        monkeypatch.setattr(
            mod.weaviate,
            "connect_to_local",
            lambda host, port, grpc_port: FakeClient(),
        )

        provider._embeddings = None
        provider._build_client(WeaviateConfig(namespace="tenant-a"))

        assert captured["use_multi_tenancy"] is True

    def test_build_client_disables_multi_tenancy_without_namespace(
        self, provider, monkeypatch
    ):
        import agent_platform.integrations.vector_store.weaviate.weaviate as mod

        captured = {}

        class FakeWeaviateVectorStore:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(mod, "WeaviateVectorStore", FakeWeaviateVectorStore)
        provider._credentials = mod.WeaviateCredentials(api_key=None)

        class FakeClient:
            pass

        monkeypatch.setattr(
            mod.weaviate,
            "connect_to_local",
            lambda host, port, grpc_port: FakeClient(),
        )

        provider._embeddings = None
        provider._build_client(WeaviateConfig())

        assert captured["use_multi_tenancy"] is False


class TestWeaviateConfigFields:
    def test_grpc_port_default(self):
        assert WeaviateConfig().grpc_port == 50051

    def test_http_host_and_port_default_to_none(self):
        cfg = WeaviateConfig()
        assert cfg.http_host is None
        assert cfg.http_port is None

    def test_structured_fields_custom(self):
        cfg = WeaviateConfig(
            http_host="weaviate.internal", http_port=9000, grpc_port=50052
        )
        assert cfg.http_host == "weaviate.internal"
        assert cfg.http_port == 9000
        assert cfg.grpc_port == 50052


class TestParseUrl:
    def test_parses_scheme_host_port(self):
        assert _parse_url("https://weaviate.example.com:8443") == (
            "weaviate.example.com",
            8443,
            True,
        )

    def test_defaults_port_when_missing(self):
        assert _parse_url("http://localhost") == ("localhost", 80, False)

    def test_https_default_port(self):
        assert _parse_url("https://weaviate.example.com") == (
            "weaviate.example.com",
            443,
            True,
        )


class TestWeaviateConnectionTarget:
    def test_prefers_structured_fields_over_url(self, provider):
        provider._credentials = type(
            "C", (), {"url": "http://localhost:8080", "api_key": None}
        )()
        host, port, secure = provider._connection_target(
            WeaviateConfig(http_host="custom-host", http_port=9999)
        )
        assert (host, port, secure) == ("custom-host", 9999, False)

    def test_falls_back_to_url_parsing_when_unset(self, provider):
        provider._credentials = type(
            "C", (), {"url": "https://weaviate.example.com:8443", "api_key": None}
        )()
        host, port, secure = provider._connection_target(WeaviateConfig())
        assert (host, port, secure) == ("weaviate.example.com", 8443, True)


class TestWeaviateConnectionLifecycle:
    def _patch_store(self, monkeypatch, mod):
        class FakeVectorStore:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def aadd_documents(self, docs):
                return None

            def delete(self, ids):
                return None

            async def asimilarity_search_by_vector(self, *a, **k):
                return []

            async def asimilarity_search_with_score(self, *a, **k):
                return []

        monkeypatch.setattr(mod, "WeaviateVectorStore", FakeVectorStore)

    async def test_search_closes_client_after_use(self, provider, monkeypatch):
        import agent_platform.integrations.vector_store.weaviate.weaviate as mod

        self._patch_store(monkeypatch, mod)

        closed = {"called": False}

        class FakeRawClient:
            def close(self):
                closed["called"] = True

        provider._credentials = mod.WeaviateCredentials(api_key=None)
        provider._embeddings = None
        monkeypatch.setattr(provider, "_connect", lambda config: FakeRawClient())

        await provider.search(query_vector=[0.1, 0.2], config=WeaviateConfig())

        assert closed["called"] is True

    async def test_add_closes_client_after_use(self, provider, monkeypatch):
        import agent_platform.integrations.vector_store.weaviate.weaviate as mod

        self._patch_store(monkeypatch, mod)

        closed = {"called": False}

        class FakeRawClient:
            def close(self):
                closed["called"] = True

        provider._credentials = mod.WeaviateCredentials(api_key=None)
        provider._embeddings = None
        monkeypatch.setattr(provider, "_connect", lambda config: FakeRawClient())

        await provider.add([TextChunk(text="hi", index=0)], config=WeaviateConfig())

        assert closed["called"] is True

    async def test_search_closes_client_even_on_failure(self, provider, monkeypatch):
        import agent_platform.core.errors as errors_mod
        import agent_platform.integrations.vector_store.weaviate.weaviate as mod

        monkeypatch.setattr(errors_mod.asyncio, "sleep", AsyncMock())

        class FailingVectorStore:
            def __init__(self, **kwargs):
                pass

            async def asimilarity_search_by_vector(self, *a, **k):
                raise ConnectionError("boom")

        monkeypatch.setattr(mod, "WeaviateVectorStore", FailingVectorStore)

        closed = {"count": 0}

        class FakeRawClient:
            def close(self):
                closed["count"] += 1

        provider._credentials = mod.WeaviateCredentials(api_key=None)
        provider._embeddings = None
        monkeypatch.setattr(provider, "_connect", lambda config: FakeRawClient())

        with pytest.raises(Exception):
            await provider.search(query_vector=[0.1, 0.2], config=WeaviateConfig())

        assert closed["count"] >= 1
