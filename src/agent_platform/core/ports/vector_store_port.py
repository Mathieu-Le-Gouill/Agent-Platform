from typing import Protocol
from core.entities.text.document import Document
from core.entities.text.chunk import Chunk
from agent_platform.core.entities.embeddings import Embeddings
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