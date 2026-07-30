from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("pinecone")

from agent_platform.core.credentials import ClientOptions, resolve_client_options
from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.credentials import PineconeCredentials
from agent_platform.integrations.vector_store.pinecone.config import PineconeConfig
from agent_platform.integrations.vector_store.pinecone.mappers import (
    chunk_to_metadata as _chunk_to_metadata,
)
from agent_platform.integrations.vector_store.pinecone.mappers import (
    match_to_chunk as _match_to_chunk,
)
from agent_platform.integrations.vector_store.pinecone.provider import PineconeStore
from tests.helpers import assert_custom_construction_stored, assert_default_construction


def _mock_index() -> MagicMock:
    index = MagicMock()
    index.__aenter__ = AsyncMock(return_value=index)
    index.__aexit__ = AsyncMock(return_value=False)
    index.upsert = AsyncMock()
    index.delete = AsyncMock()
    index.query = AsyncMock()
    return index


def test_pinecone_config_has_no_environment_field():
    assert "environment" not in PineconeConfig.model_fields


def test_pinecone_config_host_defaults_to_none():
    assert PineconeConfig().host is None


def test_pinecone_config_host_custom():
    cfg = PineconeConfig(host="my-index-abc123.svc.us-east-1.pinecone.io")
    assert cfg.host == "my-index-abc123.svc.us-east-1.pinecone.io"


class TestPineconeConstruction:
    def test_default_credentials_and_config(self):
        assert_default_construction(PineconeStore, PineconeConfig)

    def test_custom_credentials_stored(self):
        assert_custom_construction_stored(
            PineconeStore, PineconeCredentials(api_key="secret")
        )


class TestPineconeApiKey:
    def test_raises_without_api_key(self):
        store = PineconeStore.__new__(PineconeStore)
        store._credentials = PineconeCredentials(api_key=None)
        with pytest.raises(MissingCredentialError):
            store._api_key()

    def test_returns_secret_value(self):
        store = PineconeStore.__new__(PineconeStore)
        store._credentials = PineconeCredentials(api_key="secret")
        assert store._api_key() == "secret"


class TestPineconeClientKwargs:
    def test_default_has_no_timeout(self):
        store = PineconeStore.__new__(PineconeStore)
        store._credentials = PineconeCredentials(api_key="secret")
        store._client_options = resolve_client_options(None)
        kwargs = store._client_kwargs(PineconeConfig())
        assert kwargs["api_key"] == "secret"
        assert "timeout" not in kwargs

    def test_explicit_config_timeout(self):
        store = PineconeStore.__new__(PineconeStore)
        store._credentials = PineconeCredentials(api_key="secret")
        store._client_options = resolve_client_options(None)
        kwargs = store._client_kwargs(PineconeConfig(timeout=15.0))
        assert kwargs["timeout"] == 15.0

    def test_client_options_timeout_fallback(self):
        store = PineconeStore.__new__(PineconeStore)
        store._credentials = PineconeCredentials(api_key="secret")
        store._client_options = resolve_client_options(ClientOptions(timeout=60.0))
        kwargs = store._client_kwargs(PineconeConfig())
        assert kwargs["timeout"] == 60.0


class TestPineconeResolveHost:
    async def test_uses_configured_host_without_describe_call(self, mocker):
        mock_pc_cls = mocker.patch(
            "agent_platform.integrations.vector_store.pinecone.provider.PineconeAsyncio"
        )
        store = PineconeStore.__new__(PineconeStore)
        store._credentials = PineconeCredentials(api_key="secret")
        store._client_options = resolve_client_options(None)

        host = await store._resolve_host(PineconeConfig(host="idx.svc.pinecone.io"))

        assert host == "idx.svc.pinecone.io"
        mock_pc_cls.assert_not_called()

    async def test_describes_index_when_host_unset(self, mocker):
        mock_pc_cls = mocker.patch(
            "agent_platform.integrations.vector_store.pinecone.provider.PineconeAsyncio"
        )
        mock_pc = MagicMock()
        mock_pc.__aenter__ = AsyncMock(return_value=mock_pc)
        mock_pc.__aexit__ = AsyncMock(return_value=False)
        mock_pc.describe_index = AsyncMock(
            return_value=SimpleNamespace(host="resolved.svc.pinecone.io")
        )
        mock_pc_cls.return_value = mock_pc

        store = PineconeStore.__new__(PineconeStore)
        store._credentials = PineconeCredentials(api_key="secret")
        store._client_options = resolve_client_options(None)

        host = await store._resolve_host(PineconeConfig())

        assert host == "resolved.svc.pinecone.io"
        mock_pc.describe_index.assert_awaited_once()


class TestPineconeIndex:
    async def test_index_builds_from_resolved_host(self, mocker):
        mock_pc_cls = mocker.patch(
            "agent_platform.integrations.vector_store.pinecone.provider.PineconeAsyncio"
        )
        mock_pc = MagicMock()
        mock_pc.IndexAsyncio.return_value = "the-index"
        mock_pc_cls.return_value = mock_pc

        store = PineconeStore.__new__(PineconeStore)
        store._credentials = PineconeCredentials(api_key="secret")
        store._client_options = resolve_client_options(None)

        result = await store._index(PineconeConfig(host="idx.svc.pinecone.io"))

        assert result == "the-index"
        mock_pc.IndexAsyncio.assert_called_once_with(host="idx.svc.pinecone.io")


class TestPineconeMappers:
    def test_chunk_to_metadata(self):
        chunk = TextChunk(
            text="hello", index=0, metadata={"source": "doc.txt", "language": "en"}
        )
        metadata = _chunk_to_metadata(chunk)
        assert metadata["text"] == "hello"
        assert metadata["source"] == "doc.txt"
        assert metadata["language"] == "en"
        assert metadata["extra"] == "{}"

    def test_chunk_to_metadata_all_fields(self):
        chunk = TextChunk(
            text="hello", index=3, document_id=uuid4(), start_char=0, end_char=5
        )
        metadata = _chunk_to_metadata(chunk)
        assert metadata["document_id"] == str(chunk.document_id)
        assert metadata["start_char"] == 0
        assert metadata["end_char"] == 5

    def test_match_to_chunk_invalid_extra_json_falls_back_to_empty(self):
        uid = uuid4()
        match = SimpleNamespace(id=str(uid), metadata={"extra": "not json"})
        chunk = _match_to_chunk(match)
        assert chunk.metadata["extra"] == {}

    def test_match_to_chunk_round_trip(self):
        uid = uuid4()
        match = SimpleNamespace(
            id=str(uid), metadata={"text": "hello", "index": 2, "source": "doc.txt"}
        )
        chunk = _match_to_chunk(match)
        assert chunk.id == uid
        assert chunk.text == "hello"
        assert chunk.index == 2
        assert chunk.metadata["source"] == "doc.txt"

    def test_match_to_chunk_missing_metadata(self):
        uid = uuid4()
        match = SimpleNamespace(id=str(uid), metadata=None)
        chunk = _match_to_chunk(match)
        assert chunk.text == ""
        assert chunk.index == 0
        assert chunk.metadata["extra"] == {}


class TestPineconeAdd:
    async def test_upserts_vectors(self, mocker):
        mocker.patch(
            "agent_platform.integrations.vector_store.pinecone.provider.PineconeAsyncio"
        )
        store = PineconeStore.__new__(PineconeStore)
        store._credentials = PineconeCredentials(api_key="secret")
        mock_index = _mock_index()
        store._index = AsyncMock(return_value=mock_index)

        chunk = TextChunk(text="hello", index=0)
        await store.add([chunk], [[0.1, 0.2]], config=PineconeConfig())

        mock_index.upsert.assert_awaited_once()
        _, kwargs = mock_index.upsert.call_args
        assert kwargs["vectors"][0]["id"] == str(chunk.id)
        assert kwargs["vectors"][0]["values"] == [0.1, 0.2]

    async def test_add_empty_list_skips_call(self):
        store = PineconeStore.__new__(PineconeStore)
        store._index = AsyncMock()
        await store.add([], [], config=PineconeConfig())
        store._index.assert_not_called()


class TestPineconeDelete:
    async def test_deletes_by_ids(self):
        store = PineconeStore.__new__(PineconeStore)
        mock_index = _mock_index()
        store._index = AsyncMock(return_value=mock_index)

        doc_id = uuid4()
        await store.delete([doc_id], config=PineconeConfig())

        mock_index.delete.assert_awaited_once()
        _, kwargs = mock_index.delete.call_args
        assert kwargs["ids"] == [str(doc_id)]

    async def test_delete_empty_list_skips_call(self):
        store = PineconeStore.__new__(PineconeStore)
        store._index = AsyncMock()
        await store.delete([], config=PineconeConfig())
        store._index.assert_not_called()


class TestPineconeSearch:
    async def test_search_returns_chunks(self):
        store = PineconeStore.__new__(PineconeStore)
        mock_index = _mock_index()
        uid = uuid4()
        mock_index.query.return_value = SimpleNamespace(
            matches=[SimpleNamespace(id=str(uid), score=0.5, metadata={"text": "hi"})]
        )
        store._index = AsyncMock(return_value=mock_index)

        results = await store.search(query_vector=[0.1, 0.2], config=PineconeConfig())

        assert len(results) == 1
        assert results[0].id == uid
        assert results[0].text == "hi"

    async def test_search_with_scores_clamps_score(self):
        store = PineconeStore.__new__(PineconeStore)
        mock_index = _mock_index()
        uid = uuid4()
        mock_index.query.return_value = SimpleNamespace(
            matches=[SimpleNamespace(id=str(uid), score=1.5, metadata={"text": "hi"})]
        )
        store._index = AsyncMock(return_value=mock_index)

        results = await store.search_with_scores(
            query_vector=[0.1, 0.2], config=PineconeConfig()
        )

        assert results[0][1].value == 1.0

    async def test_search_with_scores_retries_transient_failure_then_succeeds(
        self, no_retry_sleep
    ):
        store = PineconeStore.__new__(PineconeStore)
        mock_index = _mock_index()
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return SimpleNamespace(matches=[])

        mock_index.query = flaky
        store._index = AsyncMock(return_value=mock_index)

        result = await store.search_with_scores(
            query_vector=[0.1, 0.2], config=PineconeConfig()
        )
        assert calls["n"] == 2
        assert result == []

    async def test_search_with_scores_translates_permanent_failure(
        self, no_retry_sleep
    ):
        store = PineconeStore.__new__(PineconeStore)
        mock_index = _mock_index()

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_index.query = always_fails
        store._index = AsyncMock(return_value=mock_index)

        with pytest.raises(ProviderError, match="Vector store search failed"):
            await store.search_with_scores(
                query_vector=[0.1, 0.2], config=PineconeConfig()
            )
