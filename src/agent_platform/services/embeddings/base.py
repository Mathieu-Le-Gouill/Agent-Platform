from typing import Protocol
from agent_platform.models.embeddable import Embeddable
from agent_platform.models.embedding import Embedding


class BaseEmbedder(Protocol):
    async def encode(
        self,
        items: list[Embeddable],
    ) -> list[Embedding]: 
        ...