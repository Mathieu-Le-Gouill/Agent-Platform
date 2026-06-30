from abc import ABC, abstractmethod
from typing import Sequence, TypeVar, Generic

from models.protocols.text_unit import TextUnit
from models.chunk import Chunk

from integrations.chunking.config import ChunkerConfig

T_contra = TypeVar("T_contra", bound=TextUnit, contravariant=True)

class BaseChunker(ABC, Generic[T_contra]):

    @abstractmethod
    async def chunk(
        self,
        documents: Sequence[T_contra],
        config: ChunkerConfig | None = None,
    ) -> list[Chunk]:
        ...