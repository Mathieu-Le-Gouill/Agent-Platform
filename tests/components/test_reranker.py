import pytest

from agent_platform.components.reranker import Reranker
from agent_platform.core.interfaces.reranking.base import BaseReranker
from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.core.schemas.chunk import TextChunk


class _FakeReranker(BaseReranker):
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[TextChunk]]] = []

    def rerank(self, query, items, config=None):
        raise NotImplementedError

    async def arerank(self, query, items, config=None):
        self.calls.append((query, list(items)))
        return list(reversed(items))


def _chunks(n: int) -> list[TextChunk]:
    return [TextChunk(text=f"doc{i}", index=i) for i in range(n)]


@pytest.mark.asyncio
async def test_single_batch_when_under_limit():
    backend = _FakeReranker()
    reranker = Reranker(backend, RerankerConfig(batch_size=10))

    result = await reranker.arun(("query", _chunks(3)))

    assert len(backend.calls) == 1
    assert backend.calls[0][0] == "query"
    assert len(result) == 3


@pytest.mark.asyncio
async def test_splits_into_multiple_batches_over_limit():
    backend = _FakeReranker()
    reranker = Reranker(backend, RerankerConfig(batch_size=2))

    chunks = _chunks(5)
    result = await reranker.arun(("query", chunks))

    assert [len(call[1]) for call in backend.calls] == [2, 2, 1]
    assert len(result) == 5


@pytest.mark.asyncio
async def test_results_concatenated_in_batch_order():
    backend = _FakeReranker()
    reranker = Reranker(backend, RerankerConfig(batch_size=2))

    chunks = _chunks(4)
    result = await reranker.arun(("query", chunks))

    # each batch of 2 is reversed by the fake backend, batches stay in order
    assert [c.text for c in result] == ["doc1", "doc0", "doc3", "doc2"]


@pytest.mark.asyncio
async def test_defaults_to_config_batch_size_when_none_given():
    backend = _FakeReranker()
    reranker = Reranker(backend, config=None)

    await reranker.arun(("query", _chunks(RerankerConfig().batch_size + 1)))

    assert len(backend.calls) == 2
