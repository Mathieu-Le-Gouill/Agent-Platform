from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI, OpenAI

from agent_platform.core.credentials import (
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import (
    ProviderError,
    error_logged,
    require_secret,
    with_retry,
)
from agent_platform.core.interfaces.embeddings.base import BaseEmbeddingProvider
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.embedding import Embedding
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.embeddings.openai.config import OpenAIEmbeddingConfig


class OpenAIEmbeddingProvider(BaseEmbeddingProvider[OpenAIEmbeddingConfig]):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, OpenAICredentials)

    def _default_config(self) -> OpenAIEmbeddingConfig:
        return OpenAIEmbeddingConfig()

    def _client_kwargs(self, config: OpenAIEmbeddingConfig) -> dict[str, Any]:
        api_key = require_secret(
            self._credentials.api_key,
            "OpenAI API key is required but was not provided",
        )
        kwargs: dict[str, Any] = {
            "api_key": api_key.get_secret_value(),
            "max_retries": resolve_max_retries(config.max_retries, self._credentials),
        }
        if self._credentials.organization is not None:
            kwargs["organization"] = self._credentials.organization

        timeout = resolve_timeout(config.timeout, self._credentials)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _client(self, config: OpenAIEmbeddingConfig) -> AsyncOpenAI:
        return AsyncOpenAI(**self._client_kwargs(config))

    def _sync_client(self, config: OpenAIEmbeddingConfig) -> OpenAI:
        return OpenAI(**self._client_kwargs(config))

    def _params(self, config: OpenAIEmbeddingConfig) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if config.dimensions is not None:
            params["dimensions"] = config.dimensions
        if config.encoding_format is not None:
            params["encoding_format"] = config.encoding_format
        params.update(config.model_kwargs or {})
        params.update(config.extra_params)
        return params

    def embed_document(
        self,
        items: Sequence[TextChunk],
        config: OpenAIEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        client = self._sync_client(config)

        texts = [item.text for item in items]
        response = client.embeddings.create(
            model=config.model, input=texts, **self._params(config)
        )

        embeddings = [
            Embedding.from_list(data.embedding, model=config.model, id=item.id)
            for item, data in zip(items, response.data)
        ]
        return EmbeddingResponse(embeddings=embeddings, model=config.model)

    @error_logged(re_raise=ProviderError, message="Embedding generation failed")
    @with_retry()
    async def aembed_document(
        self,
        items: Sequence[TextChunk],
        config: OpenAIEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        client = self._client(config)

        texts = [item.text for item in items]
        response = await client.embeddings.create(
            model=config.model, input=texts, **self._params(config)
        )

        embeddings = [
            Embedding.from_list(data.embedding, model=config.model, id=item.id)
            for item, data in zip(items, response.data)
        ]
        return EmbeddingResponse(embeddings=embeddings, model=config.model)

    def embed_query(
        self,
        query: str,
        config: OpenAIEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        client = self._sync_client(config)

        response = client.embeddings.create(
            model=config.model, input=[query], **self._params(config)
        )

        embedding = Embedding.from_list(response.data[0].embedding)
        return EmbeddingResponse(embeddings=[embedding], model=config.model)

    @error_logged(re_raise=ProviderError, message="Embedding generation failed")
    @with_retry()
    async def aembed_query(
        self,
        query: str,
        config: OpenAIEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        client = self._client(config)

        response = await client.embeddings.create(
            model=config.model, input=[query], **self._params(config)
        )

        embedding = Embedding.from_list(response.data[0].embedding)
        return EmbeddingResponse(embeddings=[embedding], model=config.model)
