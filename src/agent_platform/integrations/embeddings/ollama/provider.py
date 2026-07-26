from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ollama import AsyncClient, Client

from agent_platform.core.credentials import resolve_credentials, resolve_timeout
from agent_platform.integrations.credentials import OllamaCredentials
from agent_platform.integrations.embeddings._base import NativeEmbeddingProvider
from agent_platform.integrations.embeddings.ollama.config import OllamaEmbeddingConfig


class OllamaEmbeddingProvider(NativeEmbeddingProvider[OllamaEmbeddingConfig]):
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

    def _embed_sync(
        self, texts: list[str], config: OllamaEmbeddingConfig
    ) -> Sequence[list[float]]:
        client = self._sync_client(config)
        response = client.embed(model=config.model, input=texts, **self._params(config))
        return [list(vector) for vector in response.embeddings]

    async def _embed_async(
        self, texts: list[str], config: OllamaEmbeddingConfig
    ) -> Sequence[list[float]]:
        client = self._client(config)
        response = await client.embed(
            model=config.model, input=texts, **self._params(config)
        )
        return [list(vector) for vector in response.embeddings]
