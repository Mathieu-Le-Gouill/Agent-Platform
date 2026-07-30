from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from huggingface_hub import AsyncInferenceClient, InferenceClient
from sentence_transformers import SentenceTransformer

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_timeout,
)
from agent_platform.core.errors import require_secret
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.integrations.embeddings._base import NativeEmbeddingProvider
from agent_platform.integrations.embeddings.huggingface.config import (
    HuggingFaceEmbeddingConfig,
    HuggingFaceEmbeddingMode,
)


class HuggingFaceEmbeddingProvider(NativeEmbeddingProvider[HuggingFaceEmbeddingConfig]):
    def __init__(
        self,
        credentials: HuggingFaceCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, HuggingFaceCredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> HuggingFaceEmbeddingConfig:
        return HuggingFaceEmbeddingConfig()

    # --- local mode (sentence-transformers) ---

    def _local_client(self, config: HuggingFaceEmbeddingConfig) -> SentenceTransformer:
        kwargs: dict[str, Any] = dict(config.model_kwargs or {})
        kwargs.update(config.extra_params)
        if config.dimensions is not None:
            kwargs["truncate_dim"] = config.dimensions
        return SentenceTransformer(config.model, **kwargs)

    def _local_encode_kwargs(
        self, config: HuggingFaceEmbeddingConfig
    ) -> dict[str, Any]:
        return dict(config.encode_kwargs or {})

    def _local_encode(
        self, texts: list[str], config: HuggingFaceEmbeddingConfig
    ) -> list[list[float]]:
        vectors = self._local_client(config).encode(
            texts, **self._local_encode_kwargs(config)
        )
        return [[float(x) for x in vector] for vector in vectors]

    # --- hosted mode (Inference API) ---

    def _hosted_client_kwargs(
        self, config: HuggingFaceEmbeddingConfig
    ) -> dict[str, Any]:
        api_key = require_secret(
            self._credentials.api_key,
            "Hugging Face Hub API token is required for hosted inference",
        )
        kwargs: dict[str, Any] = {
            "model": config.model,
            "token": api_key.get_secret_value(),
        }
        if config.provider is not None:
            kwargs["provider"] = config.provider
        timeout = resolve_timeout(config.timeout, self._client_options)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _hosted_client(
        self, config: HuggingFaceEmbeddingConfig
    ) -> AsyncInferenceClient:
        return AsyncInferenceClient(**self._hosted_client_kwargs(config))

    def _hosted_sync_client(
        self, config: HuggingFaceEmbeddingConfig
    ) -> InferenceClient:
        return InferenceClient(**self._hosted_client_kwargs(config))

    def _hosted_params(self, config: HuggingFaceEmbeddingConfig) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if config.dimensions is not None:
            params["dimensions"] = config.dimensions
        if config.truncate is not None:
            params["truncate"] = config.truncate
        if config.normalize is not None:
            params["normalize"] = config.normalize
        params.update(config.model_kwargs or {})
        params.update(config.extra_params)
        return params

    def _embed_sync(
        self, texts: list[str], config: HuggingFaceEmbeddingConfig
    ) -> Sequence[list[float]]:
        if config.mode is HuggingFaceEmbeddingMode.LOCAL:
            return self._local_encode(texts, config)
        client = self._hosted_sync_client(config)
        array = client.feature_extraction(texts, **self._hosted_params(config))
        return [[float(x) for x in row] for row in array]

    async def _embed_async(
        self, texts: list[str], config: HuggingFaceEmbeddingConfig
    ) -> Sequence[list[float]]:
        if config.mode is HuggingFaceEmbeddingMode.LOCAL:
            return await asyncio.to_thread(self._local_encode, texts, config)
        client = self._hosted_client(config)
        array = await client.feature_extraction(texts, **self._hosted_params(config))
        return [[float(x) for x in row] for row in array]
