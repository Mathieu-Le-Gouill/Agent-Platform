from typing import Protocol, Sequence
from models.protocols.text_unit import TextUnit
from integrations.reranking.config import RerankerConfig


class BaseReranker(Protocol):

    async def rerank(
        self,
        items: Sequence[TextUnit],
        config: RerankerConfig | None = None,
    ) -> Sequence[TextUnit]: 
        ...