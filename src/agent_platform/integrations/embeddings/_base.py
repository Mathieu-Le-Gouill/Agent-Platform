from __future__ import annotations

import functools
from abc import abstractmethod
from collections.abc import Sequence
from typing import Generic

from agent_platform.core.errors import ProviderError, error_logged
from agent_platform.core.interfaces.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingConfigT,
)
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.embedding import Embedding
from agent_platform.integrations.sdk_errors import translate_sdk_error

_ERROR_MESSAGE = "Embedding generation failed"
_translate = functools.partial(translate_sdk_error, operation=_ERROR_MESSAGE)


class NativeEmbeddingProvider(
    BaseEmbeddingProvider[EmbeddingConfigT], Generic[EmbeddingConfigT]
):
    """Shared request/response plumbing for embedding providers calling a vendor
    SDK directly. Every subclass owns its own vendor-specific pieces (client
    construction, native call, vector extraction, including any local-vs-hosted
    branching); this base owns the `TextChunk`/query -> `EmbeddingResponse`
    shape that is otherwise identical across providers.
    """

    @abstractmethod
    def _default_config(self) -> EmbeddingConfigT: ...

    @abstractmethod
    def _embed_sync(
        self, texts: list[str], config: EmbeddingConfigT
    ) -> Sequence[list[float]]:
        """Return one raw vector per text, in order, via the blocking native call."""

    @abstractmethod
    async def _embed_async(
        self, texts: list[str], config: EmbeddingConfigT
    ) -> Sequence[list[float]]:
        """Return one raw vector per text, in order, via the async native call."""

    def embed_document(
        self,
        items: Sequence[TextChunk],
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        vectors = self._embed_sync([item.text for item in items], config)
        embeddings = [
            Embedding.from_list(vector, model=config.model, id=item.id)
            for item, vector in zip(items, vectors)
        ]
        return EmbeddingResponse(embeddings=embeddings, model=config.model)

    @error_logged(re_raise=ProviderError, message=_ERROR_MESSAGE)
    @with_retry(translate=_translate)
    async def aembed_document(
        self,
        items: Sequence[TextChunk],
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        vectors = await self._embed_async([item.text for item in items], config)
        embeddings = [
            Embedding.from_list(vector, model=config.model, id=item.id)
            for item, vector in zip(items, vectors)
        ]
        return EmbeddingResponse(embeddings=embeddings, model=config.model)

    def embed_query(
        self,
        query: str,
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        vectors = self._embed_sync([query], config)
        return EmbeddingResponse(
            embeddings=[Embedding.from_list(vectors[0])], model=config.model
        )

    @error_logged(re_raise=ProviderError, message=_ERROR_MESSAGE)
    @with_retry(translate=_translate)
    async def aembed_query(
        self,
        query: str,
        config: EmbeddingConfigT | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        vectors = await self._embed_async([query], config)
        return EmbeddingResponse(
            embeddings=[Embedding.from_list(vectors[0])], model=config.model
        )
