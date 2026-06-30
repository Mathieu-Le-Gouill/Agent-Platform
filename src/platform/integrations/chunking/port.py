from typing import Protocol, Sequence

from models.document import TextDocument
from models.chunk import Chunk

from integrations.chunking.config import ChunkerConfig


class BaseChunker(Protocol):

    async def chunk(
        self,
        documents: Sequence[TextDocument],
        config: ChunkerConfig | None = None,
    ) -> list[Chunk]:
        ...