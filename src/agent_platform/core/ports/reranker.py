from typing import Protocol
from agent_platform.core.value_objects.embeddable import Embeddable


class RerankerPort(Protocol):

    async def rerank(self, items: list[Embeddable]) -> list[Embeddable]: 
        ...