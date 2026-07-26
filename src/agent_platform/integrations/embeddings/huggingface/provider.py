from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from huggingface_hub import AsyncInferenceClient, InferenceClient
from sentence_transformers import SentenceTransformer

from agent_platform.core.credentials import resolve_credentials, resolve_timeout
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
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.integrations.embeddings.huggingface.config import (
    HuggingFaceEmbeddingConfig,
    HuggingFaceEmbeddingMode,
)


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider[HuggingFaceEmbeddingConfig]):
    def __init__(self, credentials: HuggingFaceCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, HuggingFaceCredentials)

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
        timeout = resolve_timeout(config.timeout, self._credentials)
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

    def embed_document(
        self,
        items: Sequence[TextChunk],
        config: HuggingFaceEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        texts = [item.text for item in items]

        if config.mode is HuggingFaceEmbeddingMode.LOCAL:
            vectors = self._local_encode(texts, config)
        else:
            client = self._hosted_sync_client(config)
            array = client.feature_extraction(texts, **self._hosted_params(config))
            vectors = [[float(x) for x in row] for row in array]

        embeddings = [
            Embedding.from_list(vector, model=config.model, id=item.id)
            for item, vector in zip(items, vectors)
        ]
        return EmbeddingResponse(embeddings=embeddings, model=config.model)

    @error_logged(re_raise=ProviderError, message="Embedding generation failed")
    @with_retry()
    async def aembed_document(
        self,
        items: Sequence[TextChunk],
        config: HuggingFaceEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()
        texts = [item.text for item in items]

        if config.mode is HuggingFaceEmbeddingMode.LOCAL:
            vectors = await asyncio.to_thread(self._local_encode, texts, config)
        else:
            client = self._hosted_client(config)
            array = await client.feature_extraction(
                texts, **self._hosted_params(config)
            )
            vectors = [[float(x) for x in row] for row in array]

        embeddings = [
            Embedding.from_list(vector, model=config.model, id=item.id)
            for item, vector in zip(items, vectors)
        ]
        return EmbeddingResponse(embeddings=embeddings, model=config.model)

    def embed_query(
        self,
        query: str,
        config: HuggingFaceEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()

        if config.mode is HuggingFaceEmbeddingMode.LOCAL:
            vectors = self._local_encode([query], config)
        else:
            client = self._hosted_sync_client(config)
            array = client.feature_extraction([query], **self._hosted_params(config))
            vectors = [[float(x) for x in row] for row in array]

        return EmbeddingResponse(
            embeddings=[Embedding.from_list(vectors[0])], model=config.model
        )

    @error_logged(re_raise=ProviderError, message="Embedding generation failed")
    @with_retry()
    async def aembed_query(
        self,
        query: str,
        config: HuggingFaceEmbeddingConfig | None = None,
    ) -> EmbeddingResponse:
        config = config or self._default_config()

        if config.mode is HuggingFaceEmbeddingMode.LOCAL:
            vectors = await asyncio.to_thread(self._local_encode, [query], config)
        else:
            client = self._hosted_client(config)
            array = await client.feature_extraction(
                [query], **self._hosted_params(config)
            )
            vectors = [[float(x) for x in row] for row in array]

        return EmbeddingResponse(
            embeddings=[Embedding.from_list(vectors[0])], model=config.model
        )
