from typing import Protocol
from agent_platform.core.entities.document import Document
from agent_platform.core.entities.chunk import Chunk
#from agent_platform.core.entities.embedding import Embedding
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