from typing import Protocol
from uuid import UUID

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score


class VectorStore(Protocol):
    async def add(self, documents: list[TextChunk]) -> None: ...
    async def delete(self, document_ids: list[UUID]) -> None: ...
    async def search(
        self, query_vector: list[float], k: int = 5
    ) -> list[TextChunk]: ...
    async def search_with_scores(
        self, query_vector: list[float], k: int = 5
    ) -> list[tuple[TextChunk, Score]]: ...
