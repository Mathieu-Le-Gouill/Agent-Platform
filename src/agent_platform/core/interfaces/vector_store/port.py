from typing import Any, Protocol
from uuid import UUID

from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.core.schemas.vector import SparseVector

# structural mirror of BaseVectorStore, minus its ConfigT generic: lets callers
# (components/, pipelines/) accept any provider without importing the ABC.


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
    async def search_hybrid(
        self,
        query_vector: list[float],
        sparse_vector: SparseVector,
        k: int = 5,
        config: VectorStoreConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]: ...
    async def add_hybrid(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
        config: VectorStoreConfig | None = None,
    ) -> None: ...
