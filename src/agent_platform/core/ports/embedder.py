from typing import Protocol
from core.value_objects.embeddable import Embeddable
from core.entities.embedding import Embedding


class EmbedderPort(Protocol):
    async def encode(
        self,
        items: list[Embeddable],
    ) -> list[Embedding]: 
        ...