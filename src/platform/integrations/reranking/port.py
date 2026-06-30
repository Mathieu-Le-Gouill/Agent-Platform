from typing import Protocol, Sequence, TypeVar
from models.protocols.text_unit import TextUnit
from integrations.reranking.config import RerankerConfig

T = TypeVar("T", bound=TextUnit)

class BaseReranker(Protocol[T]):

    async def rerank(
        self,
        items: Sequence[T],
        config: RerankerConfig | None = None,
    ) -> Sequence[T]: 
        ...