from abc import ABC, abstractmethod

from typing import Generic, Sequence, TypeVar

from agent_platform.core.schemas.document import Document
from agent_platform.core.schemas.chunk import Chunk

from agent_platform.core.interfaces.chunking.config import ChunkerConfig

Document_T = TypeVar("Document_T", bound=Document, contravariant=True)
Chunk_T = TypeVar("Chunk_T", bound=Chunk)
ChunkerConfigT = TypeVar("ChunkerConfigT", bound=ChunkerConfig)


class BaseChunker(ABC, Generic[Document_T, Chunk_T, ChunkerConfigT]):
    @abstractmethod
    def chunk(
        self,
        documents: Sequence[Document_T],
        config: ChunkerConfigT | None,
    ) -> list[Chunk_T]: ...
