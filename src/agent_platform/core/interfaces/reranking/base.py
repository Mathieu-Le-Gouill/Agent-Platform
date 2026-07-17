from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.interfaces.reranking.config import RerankerConfig

T = TypeVar("T", bound=TextChunk)
RerankerConfigT = TypeVar("RerankerConfigT", bound=RerankerConfig)


class BaseReranker(ABC, Generic[T, RerankerConfigT]):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        items: Sequence[T],
        config: RerankerConfigT | None = None,
    ) -> Sequence[T]: ...
