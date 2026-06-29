from typing import Protocol
from uuid import UUID

from models.chunk import Chunk
from models.document import Document
from models.score import Score


class VectorStore(Protocol):

    async def add(
        self,
        documents: list[Document],
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