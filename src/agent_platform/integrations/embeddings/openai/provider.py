from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI, OpenAI

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import require_secret
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.embeddings._base import NativeEmbeddingProvider
from agent_platform.integrations.embeddings.openai.config import OpenAIEmbeddingConfig


class OpenAIEmbeddingProvider(NativeEmbeddingProvider[OpenAIEmbeddingConfig]):
    def __init__(
        self,
        credentials: OpenAICredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, OpenAICredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> OpenAIEmbeddingConfig:
        return OpenAIEmbeddingConfig()

    def _client_kwargs(self, config: OpenAIEmbeddingConfig) -> dict[str, Any]:
        api_key = require_secret(
            self._credentials.api_key,
            "OpenAI API key is required but was not provided",
        )
        kwargs: dict[str, Any] = {
            "api_key": api_key.get_secret_value(),
            "max_retries": resolve_max_retries(
                config.max_retries, self._client_options
            ),
        }
        if self._credentials.organization is not None:
            kwargs["organization"] = self._credentials.organization

        timeout = resolve_timeout(config.timeout, self._client_options)
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

    def _embed_sync(
        self, texts: list[str], config: OpenAIEmbeddingConfig
    ) -> Sequence[list[float]]:
        client = self._sync_client(config)
        response = client.embeddings.create(
            model=config.model, input=texts, **self._params(config)
        )
        return [data.embedding for data in response.data]

    async def _embed_async(
        self, texts: list[str], config: OpenAIEmbeddingConfig
    ) -> Sequence[list[float]]:
        client = self._client(config)
        response = await client.embeddings.create(
            model=config.model, input=texts, **self._params(config)
        )
        return [data.embedding for data in response.data]
