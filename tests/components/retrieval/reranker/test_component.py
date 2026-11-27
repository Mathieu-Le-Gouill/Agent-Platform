from unittest.mock import AsyncMock

import pytest

from agent_platform.components.retrieval.reranker.component import (
    Reranker,
    RerankerInput,
)
from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.reranking.base import BaseReranker
from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import ScoreKind
from agent_platform.integrations.reranking.scoring import build_relevance_scores


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
    reranker = Reranker(backend)

    result = await reranker.arun(
        RerankerInput("query", _chunks(3), RerankerConfig(batch_size=10))
    )

    assert len(backend.calls) == 1
    assert backend.calls[0][0] == "query"
    assert len(result) == 3


class _ScoringReranker(BaseReranker):
    """Scores each chunk by its `index`, honouring the config like a real
    provider: per-call `top_k`, raw or min-max normalized scores."""

    def __init__(self) -> None:
        self.configs: list[RerankerConfig | None] = []
        self.batch_sizes: list[int] = []

    def rerank(self, query, items, config=None):
        raise NotImplementedError

    async def arerank(self, query, items, config=None):
        self.configs.append(config)
        self.batch_sizes.append(len(items))
        config = config or RerankerConfig()
        ranked = sorted(items, key=lambda c: c.index, reverse=True)
        raw = [float(c.index) for c in ranked]
        scores = build_relevance_scores(raw, normalize=config.normalize_scores)
        reranked = [
            c.model_copy(update={"confidence": s if config.return_scores else None})
            for c, s in zip(ranked, scores, strict=True)
        ]
        return reranked[: config.top_k] if config.top_k is not None else reranked


def _interleaved_chunks() -> list[TextChunk]:
    # the best chunks (highest index) are spread over different batches of 2
    return [TextChunk(text=f"doc{i}", index=i) for i in (0, 5, 1, 4, 2, 3)]


@pytest.mark.asyncio
async def test_splits_into_multiple_batches_over_limit():
    backend = _ScoringReranker()

    result = await Reranker(backend).arun(
        RerankerInput("query", _chunks(5), RerankerConfig(batch_size=2))
    )

    assert backend.batch_sizes == [2, 2, 1]
    assert len(result) == 5


@pytest.mark.asyncio
async def test_batch_size_does_not_change_the_returned_scores():
    def config(batch_size: int) -> RerankerConfig:
        return RerankerConfig(batch_size=batch_size, top_k=2, return_scores=True)

    one_batch = await Reranker(_ScoringReranker()).arun(
        RerankerInput("query", _interleaved_chunks(), config(10))
    )
    split = await Reranker(_ScoringReranker()).arun(
        RerankerInput("query", _interleaved_chunks(), config(2))
    )

    assert [(c.index, c.confidence) for c in split] == [
        (c.index, c.confidence) for c in one_batch
    ]
    assert [c.confidence.value for c in split] == [1.0, 0.8]


@pytest.mark.asyncio
async def test_missing_raw_score_in_a_batch_raises():
    backend = _FakeReranker()

    with pytest.raises(ProviderError, match="no score"):
        await Reranker(backend).arun(
            RerankerInput("query", _chunks(3), RerankerConfig(batch_size=2))
        )


@pytest.mark.asyncio
async def test_orders_results_globally_across_batches():
    reranker = Reranker(_ScoringReranker())

    result = await reranker.arun(
        RerankerInput("query", _interleaved_chunks(), RerankerConfig(batch_size=2))
    )

    assert [c.index for c in result] == [5, 4, 3, 2, 1, 0]
    assert all(c.confidence is None for c in result)


@pytest.mark.asyncio
async def test_top_k_applies_globally_across_batches():
    reranker = Reranker(_ScoringReranker())

    result = await reranker.arun(
        RerankerInput(
            "query", _interleaved_chunks(), RerankerConfig(batch_size=2, top_k=2)
        )
    )

    assert [c.index for c in result] == [5, 4]


@pytest.mark.asyncio
async def test_batches_request_every_raw_score():
    backend = _ScoringReranker()
    config = RerankerConfig(batch_size=2, top_k=2)

    await Reranker(backend).arun(RerankerInput("query", _interleaved_chunks(), config))

    assert [c.return_scores for c in backend.configs] == [True, True, True]
    assert [c.normalize_scores for c in backend.configs] == [False, False, False]
    assert [c.top_k for c in backend.configs] == [None, None, None]
    assert {c.model for c in backend.configs} == {config.model}


@pytest.mark.asyncio
async def test_normalizes_scores_over_the_merged_results():
    reranker = Reranker(_ScoringReranker())

    result = await reranker.arun(
        RerankerInput(
            "query",
            _interleaved_chunks(),
            RerankerConfig(batch_size=2, return_scores=True),
        )
    )

    assert [c.confidence.value for c in result] == [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    assert {c.confidence.kind for c in result} == {ScoreKind.RELEVANCE}


@pytest.mark.asyncio
async def test_keeps_raw_scores_when_normalization_is_off():
    reranker = Reranker(_ScoringReranker())

    result = await reranker.arun(
        RerankerInput(
            "query",
            _interleaved_chunks(),
            RerankerConfig(batch_size=2, return_scores=True, normalize_scores=False),
        )
    )

    assert [c.confidence.value for c in result] == [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]


@pytest.mark.asyncio
async def test_single_batch_forwards_the_caller_config_unchanged():
    backend = _ScoringReranker()
    config = RerankerConfig(batch_size=10, top_k=2)

    result = await Reranker(backend).arun(
        RerankerInput("query", _interleaved_chunks(), config)
    )

    assert backend.configs == [config]
    assert [c.index for c in result] == [5, 4]


@pytest.mark.asyncio
async def test_backend_error_propagates():
    backend = _FakeReranker()
    backend.arerank = AsyncMock(side_effect=ProviderError("Reranking failed"))

    with pytest.raises(ProviderError, match="Reranking failed"):
        await Reranker(backend).arun(
            RerankerInput("query", _chunks(3), RerankerConfig(batch_size=2))
        )


class _DefaultingScoringReranker(_ScoringReranker):
    def default_config(self) -> RerankerConfig:
        return RerankerConfig(model="backend-default", batch_size=2, top_k=3)


@pytest.mark.asyncio
async def test_without_config_uses_the_backend_default_and_merges_globally():
    backend = _DefaultingScoringReranker()

    result = await Reranker(backend).arun(
        RerankerInput("query", _interleaved_chunks(), None)
    )

    assert backend.batch_sizes == [2, 2, 2]
    assert {c.model for c in backend.configs} == {"backend-default"}
    assert [c.index for c in result] == [5, 4, 3]


@pytest.mark.asyncio
async def test_without_config_a_single_batch_forwards_none():
    backend = _DefaultingScoringReranker()

    await Reranker(backend).arun(RerankerInput("query", _chunks(2), None))

    assert backend.configs == [None]


@pytest.mark.asyncio
async def test_without_config_or_backend_default_makes_one_unbatched_call():
    backend = _FakeReranker()

    result = await Reranker(backend).arun(
        RerankerInput("query", _chunks(RerankerConfig().batch_size + 1), None)
    )

    assert len(backend.calls) == 1
    assert len(backend.calls[0][1]) == RerankerConfig().batch_size + 1
    assert len(result) == RerankerConfig().batch_size + 1
