from abc import ABC, abstractmethod
from uuid import UUID

from agent_platform.models.chunk import TextChunk
from agent_platform.models.score import Score


class BaseVectorStore(ABC):

    @abstractmethod
    async def add(self, documents: list[TextChunk]) -> None: ...

    @abstractmethod
    async def delete(self, document_ids: list[UUID]) -> None: ...

    @abstractmethod
    async def search(self, query_vector: list[float], k: int = 5) -> list[TextChunk]: ...

    @abstractmethod
    async def search_with_scores(self, query_vector: list[float], k: int = 5) -> list[tuple[TextChunk, Score]]: ...