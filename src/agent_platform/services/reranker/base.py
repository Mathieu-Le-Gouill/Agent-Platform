from typing import Protocol
from agent_platform.models.embeddable import Embeddable


class BaseReranker(Protocol):

    async def rerank(self, items: list[Embeddable]) -> list[Embeddable]: 
        ...