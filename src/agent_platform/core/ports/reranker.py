from typing import Protocol
from core.ports.embeddable import Embeddable


class RerankerPort(Protocol):

    async def rerank(self, items: list[Embeddable]) -> list[Embeddable]: 
        ...