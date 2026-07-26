from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mistralai import Mistral
from mistralai.models import EmbeddingResponseData

from agent_platform.core.credentials import resolve_credentials, resolve_timeout
from agent_platform.core.errors import ProviderError, require_secret
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.embeddings._base import NativeEmbeddingProvider
from agent_platform.integrations.embeddings.mistral.config import MistralEmbeddingConfig


class MistralEmbeddingProvider(NativeEmbeddingProvider[MistralEmbeddingConfig]):
    def __init__(self, credentials: MistralCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, MistralCredentials)

    def _default_config(self) -> MistralEmbeddingConfig:
        return MistralEmbeddingConfig()

    def _client(self, config: MistralEmbeddingConfig) -> Mistral:
        api_key = require_secret(
            self._credentials.api_key,
            "Mistral API key is required but was not provided",
        )
        kwargs: dict[str, Any] = {"api_key": api_key.get_secret_value()}
        if config.endpoint:
            kwargs["server_url"] = config.endpoint

        timeout = resolve_timeout(config.timeout, self._credentials)
        if timeout is not None:
            kwargs["timeout_ms"] = int(timeout * 1000)

        return Mistral(**kwargs)

    def _vector(self, data: EmbeddingResponseData) -> list[float]:
        if data.embedding is None:
            raise ProviderError("Mistral returned no embedding vector")
        return data.embedding

    def _params(self, config: MistralEmbeddingConfig) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if config.dimensions is not None:
            params["output_dimension"] = config.dimensions
        params.update(config.extra_params)
        return params

    def _embed_sync(
        self, texts: list[str], config: MistralEmbeddingConfig
    ) -> Sequence[list[float]]:
        client = self._client(config)
        response = client.embeddings.create(
            model=config.model, inputs=texts, **self._params(config)
        )
        return [self._vector(data) for data in response.data]

    async def _embed_async(
        self, texts: list[str], config: MistralEmbeddingConfig
    ) -> Sequence[list[float]]:
        client = self._client(config)
        response = await client.embeddings.create_async(
            model=config.model, inputs=texts, **self._params(config)
        )
        return [self._vector(data) for data in response.data]
