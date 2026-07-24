from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.vector_store.config import DistanceMetric
from agent_platform.integrations.vector_store.faiss.config import FAISSConfig
from agent_platform.integrations.vector_store.faiss.faiss import (
    _DISTANCE_STRATEGY_MAP,
    FAISSStore,
)


@pytest.fixture
def store():
    s = FAISSStore(MagicMock())
    s._store = MagicMock()
    return s


class TestFAISSSearchRetryAndTranslation:
    async def test_search_retries_transient_failure_then_succeeds(
        self, store, monkeypatch
    ):
        import agent_platform.core.errors as errors_mod

        monkeypatch.setattr(errors_mod.asyncio, "sleep", AsyncMock())

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
        self, store, monkeypatch
    ):
        import agent_platform.core.errors as errors_mod

        monkeypatch.setattr(errors_mod.asyncio, "sleep", AsyncMock())

        def always_fails(*args, **kwargs):
            raise ConnectionError("boom")

        store._store.similarity_search_by_vector = always_fails

        with pytest.raises(ProviderError, match="Vector store search failed"):
            await store.search(query_vector=[0.1, 0.2], config=FAISSConfig())

    async def test_search_with_scores_translates_permanent_failure(
        self, store, monkeypatch
    ):
        import agent_platform.core.errors as errors_mod

        monkeypatch.setattr(errors_mod.asyncio, "sleep", AsyncMock())

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
        import agent_platform.integrations.vector_store.faiss.faiss as mod

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
        import agent_platform.integrations.vector_store.faiss.faiss as mod

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
