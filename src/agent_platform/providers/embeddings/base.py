from typing import Protocol
from models.protocols.text_unit import TextUnit
from models.embedding import Embedding


class EmbeddingProvider(Protocol):
    async def encode(
        self,
        items: TextUnit,
    ) -> Embedding: 
        ...
        

    async def encode_batch(
        self,
        items: list[TextUnit],
    ) -> list[Embedding]: 
        ...