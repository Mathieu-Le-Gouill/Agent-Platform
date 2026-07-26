from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from uuid import UUID

from agent_platform.core.interfaces.vector_store.config import VectorStoreConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.core.schemas.vector import SparseVector

ConfigT = TypeVar("ConfigT", bound=VectorStoreConfig)


class BaseVectorStore(ABC, Generic[ConfigT]):
    @abstractmethod
    async def add(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: ConfigT | None = None,
    ) -> None: ...

    @abstractmethod
    async def delete(
        self, document_ids: list[UUID], config: ConfigT | None = None
    ) -> None: ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        config: ConfigT | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[TextChunk]: ...

    @abstractmethod
    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
        config: ConfigT | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]: ...

    async def search_hybrid(
        self,
        query_vector: list[float],
        sparse_vector: SparseVector,
        k: int = 5,
        config: ConfigT | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support hybrid (dense+sparse) search"
        )
