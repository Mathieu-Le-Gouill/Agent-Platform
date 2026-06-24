from typing import Protocol
from platform.core.entities.embedding import Embedding
from platform.core.ports.embeddable import Embeddable


class EmbedderPort(Protocol):
    async def encode(
        self,
        items: list[Embeddable],
    ) -> list[Embedding]: 
        ...