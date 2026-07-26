from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("chromadb")

from agent_platform.core.errors import ProviderError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import ChromaCredentials
from agent_platform.integrations.vector_store.chroma.config import ChromaConfig
from agent_platform.integrations.vector_store.chroma.provider import (
    ChromaStore,
    _chunk_to_metadata,
    _row_to_chunk,
)
from tests.helpers import assert_custom_construction_stored, assert_default_construction


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


class TestChromaConstruction:
    def test_default_credentials_and_config(self):
        assert_default_construction(ChromaStore, ChromaConfig)

    def test_custom_credentials_stored(self):
        assert_custom_construction_stored(
            ChromaStore, ChromaCredentials(api_key="secret")
        )


class TestChromaClient:
    def _new_store(self, api_key: str | None = None) -> ChromaStore:
        store = ChromaStore.__new__(ChromaStore)
        store._credentials = ChromaCredentials(api_key=api_key)
        return store

    async def test_persist_directory_uses_persistent_client(self, mocker):
        mock_persistent = mocker.patch(
            "agent_platform.integrations.vector_store.chroma.provider.chromadb.PersistentClient"
        )
        store = self._new_store()

        await store._client(ChromaConfig(persist_directory="/tmp/chroma"))

        _, kwargs = mock_persistent.call_args
        assert kwargs["path"] == "/tmp/chroma"

    async def test_no_persist_directory_uses_async_http_client(self, mocker):
        mock_http = mocker.patch(
            "agent_platform.integrations.vector_store.chroma.provider.chromadb.AsyncHttpClient",
            new=AsyncMock(),
        )
        store = self._new_store()

        await store._client(ChromaConfig(host="10.0.0.5", port=9000, ssl=True))

        _, kwargs = mock_http.call_args
        assert kwargs["host"] == "10.0.0.5"
        assert kwargs["port"] == 9000
        assert kwargs["ssl"] is True

    async def test_tenant_and_database_always_forwarded(self, mocker):
        mock_http = mocker.patch(
            "agent_platform.integrations.vector_store.chroma.provider.chromadb.AsyncHttpClient",
            new=AsyncMock(),
        )
        store = self._new_store()

        await store._client(ChromaConfig(tenant="acme", database="prod"))

        _, kwargs = mock_http.call_args
        assert kwargs["tenant"] == "acme"
        assert kwargs["database"] == "prod"

    async def test_api_key_uses_cloud_client(self, mocker):
        mock_cloud = mocker.patch(
            "agent_platform.integrations.vector_store.chroma.provider.chromadb.CloudClient"
        )
        store = self._new_store(api_key="cloud-key")

        await store._client(ChromaConfig())

        _, kwargs = mock_cloud.call_args
        assert kwargs["api_key"] == "cloud-key"


class TestChromaWhere:
    def test_no_filter_returns_none(self):
        store = ChromaStore.__new__(ChromaStore)
        assert store._where(None) is None

    def test_single_key_filter_passthrough(self):
        store = ChromaStore.__new__(ChromaStore)
        assert store._where({"source": "doc.txt"}) == {"source": "doc.txt"}

    def test_multi_key_filter_uses_and(self):
        store = ChromaStore.__new__(ChromaStore)
        result = store._where({"source": "doc.txt", "language": "en"})
        assert result == {"$and": [{"source": "doc.txt"}, {"language": "en"}]}


class TestChromaCollectionDispatch:
    def test_is_async_false_for_sync_client(self):
        store = ChromaStore.__new__(ChromaStore)
        assert store._is_async(MagicMock()) is False

    async def test_call_offloads_sync_function(self):
        store = ChromaStore.__new__(ChromaStore)
        result = await store._call(False, lambda x: x + 1, 41)
        assert result == 42

    async def test_call_awaits_async_function(self):
        store = ChromaStore.__new__(ChromaStore)

        async def add_one(x):
            return x + 1

        result = await store._call(True, add_one, 41)
        assert result == 42

    async def test_collection_uses_get_or_create(self):
        store = ChromaStore.__new__(ChromaStore)
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = "the-collection"

        collection, is_async = await store._collection(mock_client, ChromaConfig())

        assert collection == "the-collection"
        assert is_async is False
        mock_client.get_or_create_collection.assert_called_once_with(
            name=ChromaConfig().collection_name
        )


class TestChromaMappers:
    def test_chunk_to_metadata(self):
        chunk = TextChunk(
            text="hello", index=0, metadata={"source": "doc.txt", "language": "en"}
        )
        metadata = _chunk_to_metadata(chunk)
        assert metadata["source"] == "doc.txt"
        assert metadata["language"] == "en"
        assert metadata["extra"] == "{}"

    def test_chunk_to_metadata_all_fields(self):
        chunk = TextChunk(
            text="hello",
            index=3,
            document_id=uuid4(),
            start_char=0,
            end_char=5,
        )
        metadata = _chunk_to_metadata(chunk)
        assert metadata["document_id"] == str(chunk.document_id)
        assert metadata["index"] == 3
        assert metadata["start_char"] == 0
        assert metadata["end_char"] == 5
        assert "format" in metadata

    def test_row_to_chunk_round_trip(self):
        uid = uuid4()
        chunk = _row_to_chunk(
            str(uid), "hello", {"index": 2, "source": "doc.txt", "extra": '{"k": "v"}'}
        )
        assert chunk.id == uid
        assert chunk.text == "hello"
        assert chunk.index == 2
        assert chunk.metadata["source"] == "doc.txt"
        assert chunk.metadata["extra"] == {"k": "v"}

    def test_row_to_chunk_missing_fields(self):
        uid = uuid4()
        chunk = _row_to_chunk(str(uid), None, None)
        assert chunk.text == ""
        assert chunk.index == 0
        assert chunk.metadata["extra"] == {}

    def test_row_to_chunk_invalid_extra_json_falls_back_to_empty(self):
        uid = uuid4()
        chunk = _row_to_chunk(str(uid), "hello", {"extra": "not json"})
        assert chunk.metadata["extra"] == {}


class TestChromaAdd:
    async def test_add_calls_collection_add(self, mocker):
        mocker.patch(
            "agent_platform.integrations.vector_store.chroma.provider.chromadb.CloudClient"
        )
        store = ChromaStore.__new__(ChromaStore)
        store._credentials = ChromaCredentials(api_key="k")

        mock_collection = MagicMock()
        store._client = AsyncMock(return_value=MagicMock())
        store._collection = AsyncMock(return_value=(mock_collection, False))

        chunk = TextChunk(text="hello", index=0)
        await store.add([chunk], [[0.1, 0.2]], config=ChromaConfig())

        mock_collection.add.assert_called_once()
        _, kwargs = mock_collection.add.call_args
        assert kwargs["ids"] == [str(chunk.id)]
        assert kwargs["embeddings"] == [[0.1, 0.2]]

    async def test_add_empty_list_skips_call(self):
        store = ChromaStore.__new__(ChromaStore)
        store._client = AsyncMock()
        await store.add([], [], config=ChromaConfig())
        store._client.assert_not_called()


class TestChromaDelete:
    async def test_delete_calls_collection_delete(self):
        store = ChromaStore.__new__(ChromaStore)
        mock_collection = MagicMock()
        store._client = AsyncMock(return_value=MagicMock())
        store._collection = AsyncMock(return_value=(mock_collection, False))

        doc_id = uuid4()
        await store.delete([doc_id], config=ChromaConfig())

        mock_collection.delete.assert_called_once()
        _, kwargs = mock_collection.delete.call_args
        assert kwargs["ids"] == [str(doc_id)]

    async def test_delete_empty_list_skips_call(self):
        store = ChromaStore.__new__(ChromaStore)
        store._client = AsyncMock()
        await store.delete([], config=ChromaConfig())
        store._client.assert_not_called()


class TestChromaSearch:
    def _query_result(self, uid, distance=0.2):
        return {
            "ids": [[str(uid)]],
            "documents": [["hello"]],
            "metadatas": [[{"source": "doc.txt"}]],
            "distances": [[distance]],
        }

    async def test_search_returns_chunks(self):
        store = ChromaStore.__new__(ChromaStore)
        uid = uuid4()
        mock_collection = MagicMock()
        mock_collection.query.return_value = self._query_result(uid)
        store._client = AsyncMock(return_value=MagicMock())
        store._collection = AsyncMock(return_value=(mock_collection, False))

        results = await store.search(query_vector=[0.1, 0.2], config=ChromaConfig())

        assert len(results) == 1
        assert results[0].id == uid
        assert results[0].text == "hello"

    async def test_search_with_scores_maps_distance_to_similarity(self):
        store = ChromaStore.__new__(ChromaStore)
        uid = uuid4()
        mock_collection = MagicMock()
        mock_collection.query.return_value = self._query_result(uid, distance=0.25)
        store._client = AsyncMock(return_value=MagicMock())
        store._collection = AsyncMock(return_value=(mock_collection, False))

        results = await store.search_with_scores(
            query_vector=[0.1, 0.2], config=ChromaConfig()
        )

        assert len(results) == 1
        assert results[0][1].value == pytest.approx(0.75)

    async def test_search_with_scores_retries_transient_failure_then_succeeds(
        self, no_retry_sleep
    ):
        store = ChromaStore.__new__(ChromaStore)
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        mock_collection = MagicMock()
        mock_collection.query.side_effect = flaky
        store._client = AsyncMock(return_value=MagicMock())
        store._collection = AsyncMock(return_value=(mock_collection, False))

        result = await store.search_with_scores(
            query_vector=[0.1, 0.2], config=ChromaConfig()
        )
        assert calls["n"] == 2
        assert result == []

    async def test_search_with_scores_translates_permanent_failure(
        self, no_retry_sleep
    ):
        store = ChromaStore.__new__(ChromaStore)

        def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_collection = MagicMock()
        mock_collection.query.side_effect = always_fails
        store._client = AsyncMock(return_value=MagicMock())
        store._collection = AsyncMock(return_value=(mock_collection, False))

        with pytest.raises(ProviderError, match="Vector store search failed"):
            await store.search_with_scores(
                query_vector=[0.1, 0.2], config=ChromaConfig()
            )
