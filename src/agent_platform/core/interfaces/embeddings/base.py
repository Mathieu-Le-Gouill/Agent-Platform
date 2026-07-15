from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Generic, Sequence, TypeVar

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.schemas.chunk import TextChunk

EmbeddingConfigT = TypeVar("EmbeddingConfigT", bound=EmbeddingConfig)
CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials, covariant=True)


class BaseEmbeddingProvider(ABC, Generic[CredentialsT, EmbeddingConfigT]):
    def __init__(self, credentials: CredentialsT) -> None:
        self._credentials = credentials

    @abstractmethod
    def embed_document(
        self,
        items: Sequence[TextChunk],
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse: ...

    @abstractmethod
    async def aembed_document(
        self,
        items: Sequence[TextChunk],
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse: ...

    @abstractmethod
    def embed_query(
        self,
        query: str,
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse: ...

    @abstractmethod
    async def aembed_query(
        self,
        query: str,
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse: ...
