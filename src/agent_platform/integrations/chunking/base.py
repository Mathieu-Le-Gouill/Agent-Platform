from abc import ABC, abstractmethod
from typing import Sequence, TypeVar, Generic

from agent_platform.models.document import Document
from agent_platform.models.chunk import Chunk

from agent_platform.integrations.chunking.config import ChunkerConfig

document_T = TypeVar("document_T", bound=Document, contravariant=True)
chunk_T = TypeVar("chunk_T", bound=Chunk)

class BaseChunker(ABC, Generic[document_T, chunk_T]):

    @abstractmethod
    async def chunk(
        self,
        documents: Sequence[document_T],
        config: ChunkerConfig | None = None,
    ) -> list[chunk_T]:
        ...