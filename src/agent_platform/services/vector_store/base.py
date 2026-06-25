from typing import Protocol
from agent_platform.models.document import Document
from agent_platform.models.chunk import Chunk
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