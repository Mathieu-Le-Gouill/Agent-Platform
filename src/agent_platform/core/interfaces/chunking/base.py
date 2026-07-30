from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from agent_platform.core.interfaces.chunking.config import ChunkerConfig
from agent_platform.core.schemas.chunk import Chunk
from agent_platform.core.schemas.document import Document

DocumentT = TypeVar("DocumentT", bound=Document, contravariant=True)
ChunkT = TypeVar("ChunkT", bound=Chunk)
ChunkerConfigT = TypeVar("ChunkerConfigT", bound=ChunkerConfig)


class BaseChunker(ABC, Generic[DocumentT, ChunkT, ChunkerConfigT]):
    @abstractmethod
    def chunk(
        self,
        documents: Sequence[DocumentT],
        config: ChunkerConfigT | None = None,
    ) -> list[ChunkT]: ...
