from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence

from models.protocols.text_unit import TextUnit
from integrations.reranking.config import RerankerConfig

T = TypeVar("T", bound=TextUnit)

class BaseReranker(ABC, Generic[T]):

    @abstractmethod
    async def rerank(
        self,
        query: str,
        items: Sequence[T],
        config: RerankerConfig | None = None,
    ) -> Sequence[T]:
        ...