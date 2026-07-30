from __future__ import annotations

from typing import Any

from agent_platform.components.base import Component
from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.interfaces.vector_store.port import VectorStore
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score

VectorSearchInput = tuple[
    list[float], int, dict[str, Any] | None, VectorStoreConfig | None
]


class VectorSearch(Component[VectorSearchInput, list[tuple[TextChunk, Score]]]):
    def __init__(self, backend: VectorStore) -> None:
        self._backend = backend

    async def arun(self, input: VectorSearchInput) -> list[tuple[TextChunk, Score]]:
        vector, k, filter, config = input
        return await self._backend.search_with_scores(
            vector, k=k, config=config, filter=filter
        )
