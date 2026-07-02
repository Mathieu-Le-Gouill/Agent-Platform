from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence

from agent_platform.models.protocols.text_unit import TextUnit
from agent_platform.integrations.reranking.config import RerankerConfig

T = TypeVar("T", bound=TextUnit)

class BaseReranker(ABC, Generic[T]):

    @abstractmethod
    async def rerank(
        self,
        query: str,
        items: Sequence[T],
        config: RerankerConfig = RerankerConfig(),
    ) -> Sequence[T]:
        ...