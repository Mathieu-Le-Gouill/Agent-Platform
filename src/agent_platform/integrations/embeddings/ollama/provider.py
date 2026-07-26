from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ollama import AsyncClient, Client

from agent_platform.core.credentials import resolve_credentials, resolve_timeout
from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.embeddings.base import BaseEmbeddingProvider
from agent_platform.core.interfaces.embeddings.response import EmbeddingResponse
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.embedding import Embedding
from agent_platform.integrations.credentials import OllamaCredentials
from agent_platform.integrations.embeddings.ollama.config import OllamaEmbeddingConfig


class OllamaEmbeddingProvider(BaseEmbeddingProvider[OllamaEmbeddingConfig]):
    def __init__(self, credentials: OllamaCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, OllamaCredentials)

    def _default_config(self) -> OllamaEmbeddingConfig:
        return OllamaEmbeddingConfig()

    def _client_kwargs(self, config: OllamaEmbeddingConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"host": self._credentials.base_url}
        timeout = resolve_timeout(config.timeout, self._credentials)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _client(self, config: OllamaEmbeddingConfig) -> AsyncClient:
        return AsyncClient(**self._client_kwargs(config))

    def _sync_client(self, config: OllamaEmbeddingConfig) -> Client:
        return Client(**self._client_kwargs(config))

    def _params(self, config: OllamaEmbeddingConfig) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if config.temperature is not None:
            options["temperature"] = config.temperature
        if config.top_p is not None:
            options["top_p"] = config.top_p
        if config.top_k is not None:
            options["top_k"] = config.top_k

        params: dict[str, Any] = {}
        if options:
            params["options"] = options
        if config.dimensions is not None:
            params["dimensions"] = config.dimensions
        if config.keep_alive is not None:
            params["keep_alive"] = config.keep_alive

        params.update(config.extra_params)
        return params

    def embed_document(
        self,
        items: Sequence[TextChunk],
        config: OllamaEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        client = self._sync_client(config)

        texts = [item.text for item in items]
        response = client.embed(model=config.model, input=texts, **self._params(config))

        embeddings = [
            Embedding.from_list(list(vector), model=config.model, id=item.id)
            for item, vector in zip(items, response.embeddings)
        ]
        return EmbeddingResponse(embeddings=embeddings, model=config.model)

    @error_logged(re_raise=ProviderError, message="Embedding generation failed")
    @with_retry()
    async def aembed_document(
        self,
        items: Sequence[TextChunk],
        config: OllamaEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        client = self._client(config)

        texts = [item.text for item in items]
        response = await client.embed(
            model=config.model, input=texts, **self._params(config)
        )

        embeddings = [
            Embedding.from_list(list(vector), model=config.model, id=item.id)
            for item, vector in zip(items, response.embeddings)
        ]
        return EmbeddingResponse(embeddings=embeddings, model=config.model)

    def embed_query(
        self,
        query: str,
        config: OllamaEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        client = self._sync_client(config)

        response = client.embed(
            model=config.model, input=[query], **self._params(config)
        )

        embedding = Embedding.from_list(list(response.embeddings[0]))
        return EmbeddingResponse(embeddings=[embedding], model=config.model)

    @error_logged(re_raise=ProviderError, message="Embedding generation failed")
    @with_retry()
    async def aembed_query(
        self,
        query: str,
        config: OllamaEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        client = self._client(config)

        response = await client.embed(
            model=config.model, input=[query], **self._params(config)
        )

        embedding = Embedding.from_list(list(response.embeddings[0]))
        return EmbeddingResponse(embeddings=[embedding], model=config.model)
