from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence

from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.interfaces.reranking.config import RerankerConfig

T = TypeVar("T", bound=TextChunk)
RerankerConfigT = TypeVar("RerankerConfigT", bound=RerankerConfig)
CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)


class BaseReranker(ABC, Generic[CredentialsT, T, RerankerConfigT]):
    def __init__(self, credentials: CredentialsT) -> None:
        self._credentials = credentials

    @abstractmethod
    async def rerank(
        self,
        query: str,
        items: Sequence[T],
        config: RerankerConfigT | None = None,
    ) -> Sequence[T]: ...
