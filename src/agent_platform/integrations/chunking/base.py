from abc import ABC, abstractmethod
from typing import Sequence, TypeVar, Generic

from agent_platform.models.document import Document
from agent_platform.models.chunk import Chunk

from agent_platform.integrations.chunking.config import ChunkerConfig

Document_T = TypeVar("Document_T", bound=Document, contravariant=True)
Chunk_T = TypeVar("Chunk_T", bound=Chunk)
ChunkerConfigT = TypeVar("ChunkerConfigT", bound=ChunkerConfig)

class BaseChunker(ABC, Generic[Document_T, Chunk_T, ChunkerConfigT]):

    @abstractmethod
    async def chunk(
        self,
        documents: Sequence[Document_T],
        config: ChunkerConfigT | None = None,
    ) -> list[Chunk_T]:
        ...