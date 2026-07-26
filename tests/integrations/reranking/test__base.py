from uuid import uuid4

import pytest

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.reranking._base import NativeReranker


class _Result:
    def __init__(self, index: int, score: float | None) -> None:
        self.index = index
        self.score = score


class _FakeReranker(NativeReranker[RerankerConfig, object, object, _Result]):
    def __init__(self) -> None:
        self.sync_results: list[_Result] = []
        self.async_results: list[_Result] = []
        self.sync_calls = 0
        self.async_calls = 0

    def _default_config(self) -> RerankerConfig:
        return RerankerConfig()

    def _async_client(self, config: RerankerConfig) -> object:
        return object()

    def _invoke_sync(self, client, query, documents, config):
        self.sync_calls += 1
        return self.sync_results

    async def _invoke_async(self, client, query, documents, config):
        self.async_calls += 1
        return self.async_results

    def _result_index(self, result: _Result) -> int:
        return result.index

    def _result_score(self, result: _Result) -> float | None:
        return result.score


def _items() -> list[TextChunk]:
    return [
        TextChunk(id=uuid4(), text="a", index=0),
        TextChunk(id=uuid4(), text="b", index=1),
    ]


class TestSync:
    def test_rerank_maps_results(self):
        items = _items()
        reranker = _FakeReranker()
        reranker.sync_results = [_Result(1, 0.9), _Result(0, 0.1)]

        results = reranker.rerank("q", items)

        assert [c.text for c in results] == ["b", "a"]
        assert reranker.sync_calls == 1

    def test_rerank_empty_items_short_circuits(self):
        reranker = _FakeReranker()
        results = reranker.rerank("q", [])
        assert results == []
        assert reranker.sync_calls == 0


class TestAsync:
    async def test_arerank_maps_results(self):
        items = _items()
        reranker = _FakeReranker()
        reranker.async_results = [_Result(0, 0.5), _Result(1, 0.5)]

        results = await reranker.arerank("q", items)

        assert [c.text for c in results] == ["a", "b"]
        assert reranker.async_calls == 1

    async def test_arerank_empty_items_short_circuits(self):
        reranker = _FakeReranker()
        results = await reranker.arerank("q", [])
        assert results == []
        assert reranker.async_calls == 0

    async def test_retries_transient_failure_then_succeeds(
        self, no_retry_sleep, mocker
    ):
        items = _items()
        reranker = _FakeReranker()
        calls = {"n": 0}

        async def flaky(client, query, documents, config):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            return [_Result(0, None), _Result(1, None)]

        mocker.patch.object(reranker, "_invoke_async", side_effect=flaky)
        await reranker.arerank("q", items)

        assert calls["n"] == 2

    async def test_translates_permanent_failure_to_provider_error(
        self, no_retry_sleep, mocker
    ):
        items = _items()
        reranker = _FakeReranker()

        async def always_fails(client, query, documents, config):
            raise ConnectionError("boom")

        mocker.patch.object(reranker, "_invoke_async", side_effect=always_fails)

        with pytest.raises(ProviderError, match="Reranking failed"):
            await reranker.arerank("q", items)
