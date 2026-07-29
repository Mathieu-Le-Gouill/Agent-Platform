from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("weaviate")

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.vector_store.weaviate.config import WeaviateConfig
from agent_platform.integrations.vector_store.weaviate.mappers import (
    chunk_to_properties as _chunk_to_properties,
)
from agent_platform.integrations.vector_store.weaviate.mappers import (
    object_to_chunk as _object_to_chunk,
)
from agent_platform.integrations.vector_store.weaviate.provider import (
    WeaviateStore,
    _parse_url,
)
from tests.helpers import assert_custom_construction_stored


@pytest.fixture
def provider():
    return WeaviateStore.__new__(WeaviateStore)


def _fake_object(uuid, properties, distance=None):
    return SimpleNamespace(
        uuid=uuid,
        properties=properties,
        metadata=SimpleNamespace(distance=distance),
    )


class TestWeaviateConstruction:
    def test_default_credentials_and_config(self, monkeypatch):
        monkeypatch.delenv("WEAVIATE_URL", raising=False)
        monkeypatch.delenv("WEAVIATE_API_KEY", raising=False)
        store = WeaviateStore()
        assert store._credentials.url == "http://localhost:8080"
        assert isinstance(store._default_config(), WeaviateConfig)

    def test_custom_credentials_stored(self):
        import agent_platform.integrations.vector_store.weaviate.provider as mod

        creds = mod.WeaviateCredentials(url="http://localhost:8080", api_key=None)
        assert_custom_construction_stored(WeaviateStore, creds)


class TestWeaviateConnect:
    async def test_connect_to_local_without_api_key(self, provider, monkeypatch):
        import agent_platform.integrations.vector_store.weaviate.provider as mod

        provider._credentials = mod.WeaviateCredentials(
            url="http://localhost:8080", api_key=None
        )
        captured = {}
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        monkeypatch.setattr(
            mod.weaviate,
            "use_async_with_local",
            lambda **kw: captured.update(kw) or mock_client,
        )
        result = await provider._connect(WeaviateConfig())
        assert result is mock_client
        assert captured["host"] == "localhost"
        mock_client.connect.assert_awaited_once()

    async def test_connect_to_custom_with_api_key(self, provider, monkeypatch):
        import agent_platform.integrations.vector_store.weaviate.provider as mod

        provider._credentials = mod.WeaviateCredentials(
            url="https://weaviate.example.com:8443", api_key="secret"
        )
        captured = {}
        mock_client = MagicMock()
        mock_client.connect = AsyncMock()
        monkeypatch.setattr(
            mod.weaviate,
            "use_async_with_custom",
            lambda **kw: captured.update(kw) or mock_client,
        )
        result = await provider._connect(WeaviateConfig())
        assert result is mock_client
        assert captured["http_host"] == "weaviate.example.com"
        assert captured["http_secure"] is True


class TestWeaviateFilter:
    def test_no_filter_returns_none(self, provider):
        assert provider._filter(None) is None

    def test_single_condition_filter(self, provider):
        from weaviate.collections.classes.filters import _FilterValue

        result = provider._filter({"source": "doc.txt"})
        assert isinstance(result, _FilterValue)

    def test_multi_key_filter_combines_with_all_of(self, provider):
        from weaviate.collections.classes.filters import _Filters

        result = provider._filter({"source": "doc.txt", "language": "en"})
        assert isinstance(result, _Filters)


class TestWeaviateCollection:
    def test_with_tenant_applied_when_namespace_set(self, provider):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        provider._collection(mock_client, WeaviateConfig(namespace="tenant-a"))

        mock_collection.with_tenant.assert_called_once_with("tenant-a")

    def test_no_tenant_call_without_namespace(self, provider):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        result = provider._collection(mock_client, WeaviateConfig())

        assert result is mock_collection
        mock_collection.with_tenant.assert_not_called()


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


class TestWeaviateMappers:
    def test_chunk_to_properties(self):
        chunk = TextChunk(
            text="hello",
            index=0,
            metadata={"source": "doc.txt", "language": "en", "extra": {"k": "v"}},
        )
        props = _chunk_to_properties(chunk, "text")
        assert props["text"] == "hello"
        assert props["source"] == "doc.txt"
        assert props["language"] == "en"
        assert props["extra"] == {"k": "v"}

    def test_object_to_chunk_round_trip(self):
        uid = uuid4()
        obj = _fake_object(
            uid,
            {
                "text": "hello",
                "index": 2,
                "source": "doc.txt",
                "language": "en",
                "extra": {"k": "v"},
            },
        )
        chunk = _object_to_chunk(obj, "text")
        assert chunk.id == uid
        assert chunk.text == "hello"
        assert chunk.index == 2
        assert chunk.metadata["source"] == "doc.txt"
        assert chunk.metadata["extra"] == {"k": "v"}

    def test_object_to_chunk_missing_fields(self):
        uid = uuid4()
        obj = _fake_object(uid, {})
        chunk = _object_to_chunk(obj, "text")
        assert chunk.text == ""
        assert chunk.index == 0
        assert chunk.metadata["extra"] == {}


class TestWeaviateConnectionLifecycle:
    def _connected_client(self, monkeypatch, provider):
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        monkeypatch.setattr(provider, "_connect", AsyncMock(return_value=mock_client))
        return mock_client

    async def test_add_closes_client_after_use(self, provider, monkeypatch):
        mock_client = self._connected_client(monkeypatch, provider)
        mock_collection = MagicMock()
        mock_collection.data.insert_many = AsyncMock()
        mock_client.collections.get.return_value = mock_collection

        await provider.add(
            [TextChunk(text="hi", index=0)], [[0.1, 0.2]], config=WeaviateConfig()
        )

        mock_collection.data.insert_many.assert_awaited_once()
        mock_client.close.assert_awaited_once()

    async def test_add_skips_insert_when_empty(self, provider, monkeypatch):
        mock_client = self._connected_client(monkeypatch, provider)
        mock_collection = MagicMock()
        mock_collection.data.insert_many = AsyncMock()
        mock_client.collections.get.return_value = mock_collection

        await provider.add([], [], config=WeaviateConfig())

        mock_collection.data.insert_many.assert_not_awaited()
        mock_client.close.assert_awaited_once()

    async def test_delete_closes_client_after_use(self, provider, monkeypatch):
        mock_client = self._connected_client(monkeypatch, provider)
        mock_collection = MagicMock()
        mock_collection.data.delete_many = AsyncMock()
        mock_client.collections.get.return_value = mock_collection

        doc_id = uuid4()
        await provider.delete([doc_id], config=WeaviateConfig())

        mock_collection.data.delete_many.assert_awaited_once()
        mock_client.close.assert_awaited_once()

    async def test_search_closes_client_after_use(self, provider, monkeypatch):
        mock_client = self._connected_client(monkeypatch, provider)
        mock_collection = MagicMock()
        mock_collection.query.near_vector = AsyncMock(
            return_value=SimpleNamespace(objects=[])
        )
        mock_client.collections.get.return_value = mock_collection

        await provider.search(query_vector=[0.1, 0.2], config=WeaviateConfig())

        mock_client.close.assert_awaited_once()

    async def test_search_closes_client_even_on_failure(
        self, provider, monkeypatch, no_retry_sleep
    ):
        mock_client = self._connected_client(monkeypatch, provider)
        mock_collection = MagicMock()

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_collection.query.near_vector = always_fails
        mock_client.collections.get.return_value = mock_collection

        with pytest.raises(Exception):
            await provider.search(query_vector=[0.1, 0.2], config=WeaviateConfig())

        # `@with_retry()` reconnects and closes on each of its attempts.
        assert mock_client.close.await_count == 3

    async def test_search_with_scores_maps_results_and_closes_client(
        self, provider, monkeypatch
    ):
        mock_client = self._connected_client(monkeypatch, provider)
        mock_collection = MagicMock()
        uid = uuid4()
        obj = _fake_object(uid, {"text": "hello"}, distance=0.25)
        mock_collection.query.near_vector = AsyncMock(
            return_value=SimpleNamespace(objects=[obj])
        )
        mock_client.collections.get.return_value = mock_collection

        results = await provider.search_with_scores(
            query_vector=[0.1, 0.2], config=WeaviateConfig()
        )

        assert len(results) == 1
        assert results[0][0].text == "hello"
        assert results[0][1].value == 0.75
        mock_client.close.assert_awaited_once()
