from __future__ import annotations

from collections.abc import Sequence

from google import genai
from google.genai import types

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, require_secret
from agent_platform.integrations.credentials import GoogleCredentials
from agent_platform.integrations.embeddings._base import NativeEmbeddingProvider
from agent_platform.integrations.embeddings.google.config import GoogleEmbeddingConfig


class GoogleEmbeddingProvider(NativeEmbeddingProvider[GoogleEmbeddingConfig]):
    def __init__(self, credentials: GoogleCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, GoogleCredentials)

    def _default_config(self) -> GoogleEmbeddingConfig:
        return GoogleEmbeddingConfig()

    def _client(self) -> genai.Client:
        api_key = require_secret(
            self._credentials.api_key,
            "Google API key is required but was not provided",
        )
        return genai.Client(api_key=api_key.get_secret_value())

    def _config(self, config: GoogleEmbeddingConfig) -> types.EmbedContentConfig | None:
        if config.dimensions is None:
            return None
        return types.EmbedContentConfig(output_dimensionality=config.dimensions)

    def _vector(self, embedding: types.ContentEmbedding) -> list[float]:
        if embedding.values is None:
            raise ProviderError("Google returned no embedding vector")
        return embedding.values

    def _embed_sync(
        self, texts: list[str], config: GoogleEmbeddingConfig
    ) -> Sequence[list[float]]:
        client = self._client()
        response = client.models.embed_content(
            model=config.model,
            contents=texts,  # type: ignore[arg-type]
            config=self._config(config),
        )
        return [self._vector(e) for e in (response.embeddings or [])]

    async def _embed_async(
        self, texts: list[str], config: GoogleEmbeddingConfig
    ) -> Sequence[list[float]]:
        client = self._client()
        response = await client.aio.models.embed_content(
            model=config.model,
            contents=texts,  # type: ignore[arg-type]
            config=self._config(config),
        )
        return [self._vector(e) for e in (response.embeddings or [])]
