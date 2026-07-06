from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence

from agent_platform.models.chunk import TextChunk
from agent_platform.integrations.reranking.config import RerankerConfig

T = TypeVar("T", bound=TextChunk)
RerankerConfigT = TypeVar("RerankerConfigT", bound=RerankerConfig)

class BaseReranker(ABC, Generic[T, RerankerConfigT]):

    @abstractmethod
    async def rerank(
        self,
        query: str,
        items: Sequence[T],
        config: RerankerConfigT = RerankerConfig(),  # type: ignore[assignment]
    ) -> Sequence[T]:
        ...