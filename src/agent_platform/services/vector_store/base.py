from typing import Protocol
from models.document import Document
from models.chunk import Chunk
from typing import List


class BaseVectorStore(Protocol):
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