from abc import ABC
from uuid import UUID

from agent_platform.models.chunk import Chunk
from agent_platform.models.score import Score


class BaseVectorStore(ABC):

    async def add(
        self,
        documents: list[Chunk],
    ) -> None:
        ...


    async def delete(
        self,
        document_ids: list[UUID],
    ) -> None:
        ...


    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[Chunk]:
        ...


    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
    ) -> list[tuple[Chunk, Score]]:
        ...