from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.core.credentials import BaseCredentials
from agent_platform.core.interfaces.embeddings.base import BaseEmbeddingProvider
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.components.base import Component
from agent_platform.core.schemas.chunk import TextChunk

CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)
EmbedConfigT = TypeVar("EmbedConfigT", bound=EmbeddingConfig)


class Embedder(
    Component[list[TextChunk], EmbeddingResponse],
    Generic[CredentialsT, EmbedConfigT],
):
    def __init__(
        self,
        backend: BaseEmbeddingProvider[CredentialsT, EmbedConfigT],
        config: EmbedConfigT | None = None,
    ) -> None:
        self._backend = backend
        self._config = config

    async def arun(self, input: list[TextChunk]) -> EmbeddingResponse:
        return await self._backend.aembed_document(input, self._config)
