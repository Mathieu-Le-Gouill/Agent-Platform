from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.integrations.vector_store.faiss.config import FAISSConfig
from agent_platform.integrations.vector_store.faiss.faiss import FAISSStore


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
