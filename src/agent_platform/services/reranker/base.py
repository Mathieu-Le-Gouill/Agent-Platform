from typing import Protocol
from models.protocols.text_unit import TextUnit


class BaseReranker(Protocol):

    async def rerank(self, items: list[TextUnit]) -> list[TextUnit]: 
        ...