from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.reranking.base import BaseReranker
from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.utils.batching import chunked

RerankerConfigT = TypeVar("RerankerConfigT", bound=RerankerConfig)

ChunkT = TypeVar("ChunkT", bound=TextChunk)

RerankerInput = tuple[str, list[ChunkT], RerankerConfigT | None]


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
        query, items, config = input
        batch_size = (config or RerankerConfig()).batch_size

        results: list[ChunkT] = []
        for batch in chunked(items, batch_size):
            reranked = await self._backend.arerank(query, list(batch), config)
            results.extend(reranked)

        return results
