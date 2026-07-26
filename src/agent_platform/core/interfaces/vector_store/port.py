from typing import Any, Protocol
from uuid import UUID

from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score


class VectorStore(Protocol):
    async def add(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: VectorStoreConfig | None = None,
    ) -> None: ...
    async def delete(
        self, document_ids: list[UUID], config: VectorStoreConfig | None = None
    ) -> None: ...
    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        config: VectorStoreConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[TextChunk]: ...
    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
        config: VectorStoreConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]: ...
