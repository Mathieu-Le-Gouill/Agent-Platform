from typing import Protocol
from core.entities.document import Document
from core.entities.chunk import Chunk
from typing import List


class VectorStorePort(Protocol):
    async def add(
        self,
        documents: List[Document]
    ) -> None: 
        ...


    async def search(
        self,
        query: str
    ) -> List[Chunk]:
        ...