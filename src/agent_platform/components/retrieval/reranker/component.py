from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.reranking.base import BaseReranker
from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.reranking.scoring import build_relevance_scores
from agent_platform.utils.batching import chunked

RerankerConfigT = TypeVar("RerankerConfigT", bound=RerankerConfig)

ChunkT = TypeVar("ChunkT", bound=TextChunk)


class RerankerInput(NamedTuple, Generic[ChunkT, RerankerConfigT]):
    query: str
    items: list[ChunkT]
    config: RerankerConfigT | None


class Reranker(
    Component[RerankerInput[ChunkT, RerankerConfigT], list[ChunkT]],
    Generic[ChunkT, RerankerConfigT],
):
    def __init__(
        self,
        backend: BaseReranker[ChunkT, RerankerConfigT],
    ) -> None:
        self._backend = backend

    async def arun(self, input: RerankerInput[ChunkT, RerankerConfigT]) -> list[ChunkT]:
        query, items, requested = input
        config = requested
        if config is None:
            try:
                config = self._backend.default_config()
            except NotImplementedError:
                # No provider-specific default to request scores with, so
                # batches could not be merged: one call ranks every item.
                return list(await self._backend.arerank(query, items, None))
        if len(items) <= config.batch_size:
            return list(await self._backend.arerank(query, items, requested))

        # Batches are merged on the raw provider scores: min-max normalized
        # ones are relative to their own batch, not comparable across batches.
        # Every item is scored and normalized before the global top_k cut. A
        # single batch forwards the caller's config instead, and providers that
        # cut to top_k server-side (cohere, jina, voyage) then normalize over
        # their top_k results only, so normalized scores can differ by batch_size.
        batch_config = config.model_copy(
            update={"return_scores": True, "normalize_scores": False, "top_k": None}
        )
        scored: list[tuple[ChunkT, float]] = []
        for batch in chunked(items, config.batch_size):
            reranked = await self._backend.arerank(query, list(batch), batch_config)
            scored.extend((chunk, _raw_score(chunk)) for chunk in reranked)

        scored.sort(key=lambda pair: pair[1], reverse=True)
        scores: list[Score | None] = [None] * len(scored)
        if config.return_scores:
            scores = build_relevance_scores(
                [raw for _, raw in scored], normalize=config.normalize_scores
            )
        merged = [
            chunk.model_copy(update={"confidence": score})
            for (chunk, _), score in zip(scored, scores, strict=True)
        ]
        return merged if config.top_k is None else merged[: config.top_k]


def _raw_score(chunk: TextChunk) -> float:
    if chunk.confidence is None:
        raise ProviderError(
            "Reranker backend returned no score for a chunk although "
            "return_scores=True was requested"
        )
    return chunk.confidence.value
