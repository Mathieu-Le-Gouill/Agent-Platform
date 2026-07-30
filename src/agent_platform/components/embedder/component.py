from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.embeddings.base import BaseEmbeddingProvider
from agent_platform.core.interfaces.embeddings.config import EmbeddingConfig
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.utils.batching import chunked

EmbedConfigT = TypeVar("EmbedConfigT", bound=EmbeddingConfig)

EmbedderInput = tuple[list[TextChunk], EmbedConfigT | None]


class Embedder(
    Component[EmbedderInput[EmbedConfigT], EmbeddingResponse],
    Generic[EmbedConfigT],
):
    def __init__(
        self,
        backend: BaseEmbeddingProvider[EmbedConfigT],
    ) -> None:
        self._backend = backend

    async def arun(self, input: EmbedderInput[EmbedConfigT]) -> EmbeddingResponse:
        chunks, config = input
        batch_size = (config or EmbeddingConfig()).batch_size

        embeddings = []
        model = ""
        for batch in chunked(chunks, batch_size):
            response = await self._backend.aembed_document(list(batch), config)
            embeddings.extend(response.embeddings)
            model = response.model

        return EmbeddingResponse(embeddings=embeddings, model=model)
