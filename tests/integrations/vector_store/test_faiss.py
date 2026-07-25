from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.vector_store.config import DistanceMetric
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.vector_store.faiss.config import FAISSConfig
from agent_platform.integrations.vector_store.faiss.provider import (
    _DISTANCE_STRATEGY_MAP,
    FAISSStore,
)


@pytest.fixture
def store():
    s = FAISSStore(MagicMock())
    s._store = MagicMock()
    return s


class TestFAISSDefaultConfig:
    def test_default_config(self):
        store = FAISSStore()
        assert isinstance(store._default_config(), FAISSConfig)


class TestFAISSLoadOrNone:
    def test_no_index_path_returns_none(self):
        store = FAISSStore()
        assert store._load_or_none(FAISSConfig()) is None

    def test_existing_store_returned_without_loading(self):
        existing = MagicMock()
        store = FAISSStore()
        store._store = existing
        assert store._load_or_none(FAISSConfig(index_path="/nonexistent")) is existing

    def test_missing_embeddings_raises_when_index_path_exists(self, tmp_path):
        index_path = tmp_path / "index"
        index_path.mkdir()
        store = FAISSStore(embeddings=None)
        with pytest.raises(ProviderError, match="Embeddings are required to load"):
            store._load_or_none(FAISSConfig(index_path=str(index_path)))


class TestFAISSAdd:
    async def test_missing_embeddings_raises(self):
        store = FAISSStore(embeddings=None)
        with pytest.raises(
            ProviderError, match="Embeddings are required to initialize"
        ):
            await store.add([TextChunk(text="hello", index=0)])

    async def test_creates_new_store_when_none(self, monkeypatch):
        import agent_platform.integrations.vector_store.faiss.provider as mod

        new_store = MagicMock()
        monkeypatch.setattr(mod.FAISS, "from_documents", lambda *a, **kw: new_store)

        store = FAISSStore(embeddings=MagicMock())
        await store.add([TextChunk(text="hello", index=0)])

        assert store._store is new_store

    async def test_appends_to_existing_store(self):
        existing = MagicMock()
        existing.aadd_documents = AsyncMock()
        store = FAISSStore(embeddings=MagicMock())
        store._store = existing

        await store.add([TextChunk(text="hello", index=0)])

        existing.aadd_documents.assert_awaited_once()

    async def test_saves_to_index_path_when_configured(self, tmp_path):
        existing = MagicMock()
        existing.aadd_documents = AsyncMock()
        store = FAISSStore(embeddings=MagicMock())
        store._store = existing

        index_path = str(tmp_path / "idx")
        await store.add(
            [TextChunk(text="hello", index=0)],
            config=FAISSConfig(index_path=index_path),
        )

        existing.save_local.assert_called_once_with(index_path)


class TestFAISSDelete:
    async def test_no_store_returns_without_error(self):
        store = FAISSStore()
        await store.delete([])

    async def test_deletes_from_existing_store(self):
        existing = MagicMock()
        store = FAISSStore()
        store._store = existing
        doc_id = __import__("uuid").uuid4()

        await store.delete([doc_id])

        existing.delete.assert_called_once_with([str(doc_id)])


class TestFAISSSearchNoStore:
    async def test_search_returns_empty_when_no_store(self):
        store = FAISSStore()
        result = await store.search(query_vector=[0.1], config=FAISSConfig())
        assert result == []

    async def test_search_with_scores_returns_empty_when_no_store(self):
        store = FAISSStore()
        result = await store.search_with_scores(
            query_vector=[0.1], config=FAISSConfig()
        )
        assert result == []


class TestFAISSSearchWithScoresMapping:
    async def test_maps_and_clamps_scores(self, store):
        from agent_platform.integrations.vector_store.langchain_base import _chunk_to_lc

        doc = _chunk_to_lc(TextChunk(text="hello", index=0))
        store._store.asimilarity_search_with_score_by_vector = AsyncMock(
            return_value=[(doc, 1.5), (doc, -0.5)]
        )

        results = await store.search_with_scores(
            query_vector=[0.1], config=FAISSConfig()
        )

        assert len(results) == 2
        assert results[0][1].value == 1.0
        assert results[1][1].value == 0.0


class TestFAISSSearchRetryAndTranslation:
    async def test_search_retries_transient_failure_then_succeeds(
        self, store, no_retry_sleep
    ):
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return []

        store._store.similarity_search_by_vector = flaky

        await store.search(query_vector=[0.1, 0.2], config=FAISSConfig())

        assert calls["n"] == 2

    async def test_search_translates_permanent_failure_to_provider_error(
        self, store, no_retry_sleep
    ):
        def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        store._store.similarity_search_by_vector = always_fails

        with pytest.raises(ProviderError, match="Vector store search failed"):
            await store.search(query_vector=[0.1, 0.2], config=FAISSConfig())

    async def test_search_with_scores_translates_permanent_failure(
        self, store, no_retry_sleep
    ):
        def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        store._store.similarity_search_by_vector_with_relevance_scores = always_fails

        with pytest.raises(ProviderError, match="Vector store search failed"):
            await store.search_with_scores(
                query_vector=[0.1, 0.2], config=FAISSConfig()
            )


class TestFAISSDistanceStrategy:
    def test_distance_strategy_map_covers_all_metrics(self):
        from langchain_community.vectorstores.utils import DistanceStrategy

        assert _DISTANCE_STRATEGY_MAP[DistanceMetric.COSINE] == DistanceStrategy.COSINE
        assert (
            _DISTANCE_STRATEGY_MAP[DistanceMetric.EUCLIDEAN]
            == DistanceStrategy.EUCLIDEAN_DISTANCE
        )
        assert (
            _DISTANCE_STRATEGY_MAP[DistanceMetric.DOT] == DistanceStrategy.DOT_PRODUCT
        )

    async def test_add_passes_distance_strategy_to_from_documents(self, monkeypatch):
        import agent_platform.integrations.vector_store.faiss.provider as mod

        captured = {}

        def fake_from_documents(*args, **kwargs):
            captured["distance_strategy"] = kwargs.get("distance_strategy")
            return MagicMock()

        monkeypatch.setattr(mod.FAISS, "from_documents", fake_from_documents)

        store = FAISSStore(MagicMock())
        store._store = None

        await store.add(
            [],
            config=FAISSConfig(distance=DistanceMetric.EUCLIDEAN),
        )

        from langchain_community.vectorstores.utils import DistanceStrategy

        assert captured["distance_strategy"] == DistanceStrategy.EUCLIDEAN_DISTANCE

    async def test_load_local_passes_distance_strategy(self, monkeypatch, tmp_path):
        import agent_platform.integrations.vector_store.faiss.provider as mod

        captured = {}
        index_path = tmp_path / "index"
        index_path.mkdir()

        def fake_load_local(*args, **kwargs):
            captured["distance_strategy"] = kwargs.get("distance_strategy")
            return MagicMock()

        monkeypatch.setattr(mod.FAISS, "load_local", fake_load_local)

        store = FAISSStore(MagicMock())

        store._load_or_none(
            FAISSConfig(index_path=str(index_path), distance=DistanceMetric.DOT)
        )

        from langchain_community.vectorstores.utils import DistanceStrategy

        assert captured["distance_strategy"] == DistanceStrategy.DOT_PRODUCT
