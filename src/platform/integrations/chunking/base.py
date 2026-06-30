from abc import ABC, abstractmethod
from typing import Sequence

from models.document import TextDocument
from models.chunk import Chunk

from integrations.chunking.config import ChunkerConfig


class BaseChunker(ABC):


    @abstractmethod
    async def chunk(
        self,
        documents: Sequence[TextDocument],
        config: ChunkerConfig | None = None,
    ) -> list[Chunk]:
        ...