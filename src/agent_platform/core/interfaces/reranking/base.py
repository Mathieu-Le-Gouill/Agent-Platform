from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.core.schemas.chunk import TextChunk

ChunkT = TypeVar("ChunkT", bound=TextChunk)
RerankerConfigT = TypeVar("RerankerConfigT", bound=RerankerConfig)


class BaseReranker(ABC, Generic[ChunkT, RerankerConfigT]):
    @abstractmethod
    def rerank(
        self,
        query: str,
        items: Sequence[ChunkT],
        config: RerankerConfigT | None = None,
    ) -> Sequence[ChunkT]: ...

    @abstractmethod
    async def arerank(
        self,
        query: str,
        items: Sequence[ChunkT],
        config: RerankerConfigT | None = None,
    ) -> Sequence[ChunkT]: ...
