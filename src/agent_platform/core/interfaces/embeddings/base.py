from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk

EmbeddingConfigT = TypeVar("EmbeddingConfigT", bound=EmbeddingConfig)


class BaseEmbeddingProvider(ABC, Generic[EmbeddingConfigT]):
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
