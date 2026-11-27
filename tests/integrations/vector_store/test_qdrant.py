from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("qdrant_client")

from agent_platform.core.errors import ConfigError, ProviderError
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.vector import SparseVector
from agent_platform.integrations.credentials import QdrantCredentials
from agent_platform.integrations.vector_store.qdrant.config import QdrantConfig
from agent_platform.integrations.vector_store.qdrant.mappers import (
    chunk_to_payload as _chunk_to_payload,
)
from agent_platform.integrations.vector_store.qdrant.mappers import (
    point_to_chunk as _point_to_chunk,
)
from agent_platform.integrations.vector_store.qdrant.provider import (
    QdrantVectorStoreProvider,
)
from tests.helpers import assert_custom_construction_stored, assert_default_construction


@pytest.fixture
def provider():
    return QdrantVectorStoreProvider.__new__(QdrantVectorStoreProvider)


def _scored_point(uid, payload, score=0.9):
    return SimpleNamespace(id=str(uid), payload=payload, score=score)


def test_qdrant_config_has_no_api_key_field():
    assert "api_key" not in QdrantConfig.model_fields


def test_qdrant_config_prefer_grpc_defaults_false():
    assert QdrantConfig().prefer_grpc is False


def test_qdrant_config_prefer_grpc_custom():
    assert QdrantConfig(prefer_grpc=True).prefer_grpc is True


class TestQdrantConstruction:
    def test_default_credentials_and_config(self):
        assert_default_construction(QdrantVectorStoreProvider, QdrantConfig)

    def test_custom_credentials_stored(self):
        assert_custom_construction_stored(
            QdrantVectorStoreProvider, QdrantCredentials(api_key="secret")
        )


class TestQdrantClient:
    def test_client_forwards_prefer_grpc(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        provider._client(QdrantConfig(prefer_grpc=True))
        _, kwargs = mock_client_cls.call_args
        assert kwargs["prefer_grpc"] is True

    def test_client_uses_credentials_api_key(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key="topsecret"))
        provider._client(QdrantConfig())
        _, kwargs = mock_client_cls.call_args
        assert kwargs["api_key"] == "topsecret"

    def test_client_no_api_key(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        provider._client(QdrantConfig())
        _, kwargs = mock_client_cls.call_args
        assert kwargs["api_key"] is None

    def test_client_default_timeout_is_none(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        provider._client(QdrantConfig())
        _, kwargs = mock_client_cls.call_args
        assert kwargs["timeout"] is None

    def test_client_config_timeout_cast_to_int(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        provider._client(QdrantConfig(timeout=12.7))
        _, kwargs = mock_client_cls.call_args
        assert kwargs["timeout"] == 12

    def test_client_options_timeout_fallback(self, mocker):
        from agent_platform.core.credentials import ClientOptions

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        provider = QdrantVectorStoreProvider(
            QdrantCredentials(api_key=None),
            client_options=ClientOptions(timeout=30.0),
        )
        provider._client(QdrantConfig())
        _, kwargs = mock_client_cls.call_args
        assert kwargs["timeout"] == 30


class TestQdrantFilter:
    def test_no_filter_returns_none(self, provider):
        assert provider._filter(None) is None

    def test_empty_filter_returns_none(self, provider):
        assert provider._filter({}) is None

    def test_filter_builds_qdrant_filter_model(self, provider):
        from qdrant_client import models

        qfilter = provider._filter({"source": "doc.txt"})
        assert isinstance(qfilter, models.Filter)
        assert len(qfilter.must) == 1
        condition = qfilter.must[0]
        assert isinstance(condition, models.FieldCondition)
        assert condition.key == "source"
        assert condition.match == models.MatchValue(value="doc.txt")

    def test_multi_key_filter_produces_multiple_conditions(self, provider):
        qfilter = provider._filter({"source": "doc.txt", "language": "en"})
        assert len(qfilter.must) == 2


class TestQdrantMappers:
    def test_chunk_to_payload(self):
        chunk = TextChunk(
            text="hello", index=0, metadata={"source": "doc.txt", "language": "en"}
        )
        payload = _chunk_to_payload(chunk)
        assert payload["text"] == "hello"
        assert payload["source"] == "doc.txt"
        assert payload["language"] == "en"

    def test_point_to_chunk_round_trip(self):
        uid = uuid4()
        point = _scored_point(uid, {"text": "hello", "index": 2, "source": "doc.txt"})
        chunk = _point_to_chunk(point)
        assert chunk.id == uid
        assert chunk.text == "hello"
        assert chunk.index == 2
        assert chunk.metadata["source"] == "doc.txt"

    def test_point_to_chunk_missing_payload_fields(self):
        uid = uuid4()
        point = _scored_point(uid, {})
        chunk = _point_to_chunk(point)
        assert chunk.text == ""
        assert chunk.index == 0
        assert chunk.metadata["extra"] == {}


class TestQdrantAdd:
    async def test_upserts_points(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.upsert = AsyncMock()

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        chunk = TextChunk(text="hello", index=0)
        await provider.add([chunk], [[0.1, 0.2]], config=QdrantConfig())

        mock_client.upsert.assert_awaited_once()
        _, kwargs = mock_client.upsert.call_args
        assert kwargs["collection_name"] == QdrantConfig().collection_name
        assert kwargs["points"][0].id == str(chunk.id)

    async def test_add_empty_list_skips_upsert(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.upsert = AsyncMock()

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        await provider.add([], [], config=QdrantConfig())

        mock_client.upsert.assert_not_awaited()


class TestQdrantDelete:
    async def test_deletes_by_ids(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.delete = AsyncMock()

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        doc_id = uuid4()
        await provider.delete([doc_id], config=QdrantConfig())

        mock_client.delete.assert_awaited_once()
        _, kwargs = mock_client.delete.call_args
        assert kwargs["points_selector"].points == [str(doc_id)]

    async def test_delete_empty_list_skips_call(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.delete = AsyncMock()

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        await provider.delete([], config=QdrantConfig())

        mock_client.delete.assert_not_awaited()


class TestQdrantSearch:
    async def test_search_returns_chunks(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        uid = uuid4()
        point = _scored_point(uid, {"text": "hello"}, score=0.5)
        mock_client.query_points = AsyncMock(
            return_value=SimpleNamespace(points=[point])
        )

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        results = await provider.search(query_vector=[0.1, 0.2], config=QdrantConfig())

        assert len(results) == 1
        assert results[0].id == uid
        assert results[0].text == "hello"

    async def test_search_with_scores_clamps_score(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        uid = uuid4()
        point = _scored_point(uid, {"text": "hello"}, score=1.5)
        mock_client.query_points = AsyncMock(
            return_value=SimpleNamespace(points=[point])
        )

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        results = await provider.search_with_scores(
            query_vector=[0.1, 0.2], config=QdrantConfig()
        )

        assert len(results) == 1
        assert results[0][1].value == 1.0

    async def test_search_with_scores_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        calls = {"n": 0}

        async def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return SimpleNamespace(points=[])

        mock_client.query_points = flaky

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        result = await provider.search_with_scores(
            query_vector=[0.1, 0.2], config=QdrantConfig()
        )
        assert calls["n"] == 2
        assert result == []

    async def test_search_with_scores_translates_permanent_failure(
        self, mocker, no_retry_sleep
    ):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.query_points = always_fails

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        with pytest.raises(ProviderError, match="Vector store search failed"):
            await provider.search_with_scores(
                query_vector=[0.1, 0.2], config=QdrantConfig()
            )


class TestQdrantWriteErrors:
    def _provider(self, mocker) -> tuple[QdrantVectorStoreProvider, MagicMock]:
        mock_client = MagicMock()
        mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient",
            return_value=mock_client,
        )
        return QdrantVectorStoreProvider(QdrantCredentials(api_key=None)), mock_client

    async def test_add_retries_transient_failure_then_succeeds(
        self, mocker, no_retry_sleep
    ):
        provider, client = self._provider(mocker)
        client.upsert = AsyncMock(side_effect=[ConnectionError("transient"), None])

        await provider.add([TextChunk(text="a", index=0)], [[0.1]], QdrantConfig())

        assert client.upsert.await_count == 2

    async def test_delete_translates_permanent_failure(self, mocker, no_retry_sleep):
        provider, client = self._provider(mocker)
        client.delete = AsyncMock(side_effect=ValueError("bad id"))

        with pytest.raises(ProviderError, match="Vector store delete failed"):
            await provider.delete([uuid4()], QdrantConfig())
        assert client.delete.await_count == 1

    async def test_add_hybrid_translates_failure(self, mocker, no_retry_sleep):
        provider, client = self._provider(mocker)
        client.upsert = AsyncMock(side_effect=ConnectionError("down"))

        with pytest.raises(ProviderError, match="Vector store hybrid add failed"):
            await provider.add_hybrid(
                [TextChunk(text="a", index=0)],
                [[0.1]],
                [SparseVector(indices=[1], values=[1.0])],
                QdrantConfig(),
            )
        assert client.upsert.await_count == 3

    async def test_add_hybrid_rejects_missing_sparse_vector(
        self, mocker, no_retry_sleep
    ):
        provider, client = self._provider(mocker)
        client.upsert = AsyncMock()

        with pytest.raises(ProviderError, match="Vector store hybrid add failed"):
            await provider.add_hybrid(
                [TextChunk(text="a", index=0), TextChunk(text="b", index=1)],
                [[0.1], [0.2]],
                [SparseVector(indices=[1], values=[1.0])],
                QdrantConfig(),
            )
        client.upsert.assert_not_awaited()

    async def test_add_rejects_more_vectors_than_documents(
        self, mocker, no_retry_sleep
    ):
        provider, client = self._provider(mocker)
        client.upsert = AsyncMock()

        with pytest.raises(ProviderError, match="Vector store add failed"):
            await provider.add(
                [TextChunk(text="a", index=0)], [[0.1], [0.2]], QdrantConfig()
            )
        client.upsert.assert_not_awaited()


class TestQdrantAddHybrid:
    async def test_upserts_named_vectors(self, mocker):
        from agent_platform.core.schemas.vector import SparseVector

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.upsert = AsyncMock()

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        chunk = TextChunk(text="hello", index=0)
        sparse = SparseVector(indices=[1, 3], values=[2.0, 1.0])
        await provider.add_hybrid(
            [chunk], [[0.1, 0.2]], [sparse], config=QdrantConfig()
        )

        mock_client.upsert.assert_awaited_once()
        _, kwargs = mock_client.upsert.call_args
        point = kwargs["points"][0]
        assert point.vector["dense"] == [0.1, 0.2]
        assert point.vector["sparse"].indices == [1, 3]
        assert point.vector["sparse"].values == [2.0, 1.0]

    async def test_add_hybrid_empty_list_skips_upsert(self, mocker):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.upsert = AsyncMock()

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        await provider.add_hybrid([], [], [], config=QdrantConfig())

        mock_client.upsert.assert_not_awaited()


class TestQdrantSearchHybrid:
    async def test_search_hybrid_fuses_prefetch(self, mocker):
        from agent_platform.core.schemas.vector import SparseVector

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        uid = uuid4()
        point = _scored_point(uid, {"text": "hello"}, score=0.5)
        mock_client.query_points = AsyncMock(
            return_value=SimpleNamespace(points=[point])
        )

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        sparse = SparseVector(indices=[1], values=[1.0])
        results = await provider.search_hybrid(
            query_vector=[0.1, 0.2], sparse_vector=sparse, config=QdrantConfig()
        )

        assert len(results) == 1
        assert results[0][0].id == uid
        _, kwargs = mock_client.query_points.call_args
        assert len(kwargs["prefetch"]) == 2
        assert kwargs["prefetch"][0].using == "dense"
        assert kwargs["prefetch"][1].using == "sparse"

    async def test_search_hybrid_translates_permanent_failure(
        self, mocker, no_retry_sleep
    ):
        from agent_platform.core.schemas.vector import SparseVector

        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        async def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        mock_client.query_points = always_fails

        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))
        sparse = SparseVector(indices=[1], values=[1.0])
        with pytest.raises(ProviderError, match="Vector store hybrid search failed"):
            await provider.search_hybrid(
                query_vector=[0.1, 0.2], sparse_vector=sparse, config=QdrantConfig()
            )


class TestQdrantRejectsNamespace:
    @pytest.mark.parametrize(
        "call",
        [
            lambda p, c: p.add([TextChunk(text="a", index=0)], [[0.1]], config=c),
            lambda p, c: p.add([], [], config=c),
            lambda p, c: p.delete([uuid4()], config=c),
            lambda p, c: p.search(query_vector=[0.1], config=c),
            lambda p, c: p.search_with_scores(query_vector=[0.1], config=c),
            lambda p, c: p.add_hybrid(
                [TextChunk(text="a", index=0)],
                [[0.1]],
                [SparseVector(indices=[1], values=[1.0])],
                config=c,
            ),
            lambda p, c: p.search_hybrid(
                query_vector=[0.1],
                sparse_vector=SparseVector(indices=[1], values=[1.0]),
                config=c,
            ),
        ],
        ids=[
            "add",
            "add_empty",
            "delete",
            "search",
            "search_with_scores",
            "add_hybrid",
            "search_hybrid",
        ],
    )
    async def test_operation_raises_before_any_client_call(self, mocker, call):
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))

        with pytest.raises(
            ConfigError, match="Qdrant does not support namespace isolation"
        ):
            await call(provider, QdrantConfig(namespace="tenant-a"))

        mock_client_cls.assert_not_called()

    async def test_base_config_from_a_pipeline_is_rejected_too(self, mocker):
        # rag pipelines pass a plain VectorStoreConfig, not a QdrantConfig
        mock_client_cls = mocker.patch(
            "agent_platform.integrations.vector_store.qdrant.provider.AsyncQdrantClient"
        )
        provider = QdrantVectorStoreProvider(QdrantCredentials(api_key=None))

        with pytest.raises(ConfigError):
            await provider.add(
                [TextChunk(text="a", index=0)],
                [[0.1]],
                config=VectorStoreConfig(namespace="tenant-a"),  # type: ignore[arg-type]
            )

        mock_client_cls.assert_not_called()
