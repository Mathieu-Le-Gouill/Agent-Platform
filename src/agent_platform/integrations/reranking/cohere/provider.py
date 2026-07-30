from __future__ import annotations

from typing import Any

import cohere

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import require_secret
from agent_platform.integrations.credentials import CohereCredentials
from agent_platform.integrations.reranking._base import NativeReranker
from agent_platform.integrations.reranking.cohere.config import CohereRerankerConfig


class CohereRerankerProvider(
    NativeReranker[CohereRerankerConfig, cohere.AsyncClient, cohere.Client, Any]
):
    def __init__(
        self,
        credentials: CohereCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, CohereCredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> CohereRerankerConfig:
        return CohereRerankerConfig()

    def _client_kwargs(self, config: CohereRerankerConfig) -> dict[str, Any]:
        api_key = require_secret(
            self._credentials.api_key, "Cohere API key is required"
        )
        kwargs: dict[str, Any] = {
            "api_key": api_key.get_secret_value(),
            "base_url": self._client_options.base_url,
            "max_retries": resolve_max_retries(
                config.max_retries, self._client_options
            ),
        }
        timeout = resolve_timeout(config.timeout, self._client_options)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _async_client(self, config: CohereRerankerConfig) -> cohere.AsyncClient:
        return cohere.AsyncClient(**self._client_kwargs(config))

    def _sync_client(self, config: CohereRerankerConfig) -> cohere.Client:
        return cohere.Client(**self._client_kwargs(config))

    def _rerank_kwargs(self, config: CohereRerankerConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if config.max_chunks_per_doc is not None:
            kwargs["max_chunks_per_doc"] = config.max_chunks_per_doc
        return kwargs

    def _invoke_sync(
        self,
        client: cohere.Client,
        query: str,
        documents: list[str],
        config: CohereRerankerConfig,
    ) -> Any:
        response = client.rerank(
            model=config.model,
            query=query,
            documents=documents,
            # `top_n` defaults to 3 server-side, which would silently
            # truncate results before our own top_k slice ever runs.
            top_n=config.top_k,
            **self._rerank_kwargs(config),
        )
        return response.results

    async def _invoke_async(
        self,
        client: cohere.AsyncClient,
        query: str,
        documents: list[str],
        config: CohereRerankerConfig,
    ) -> Any:
        response = await client.rerank(
            model=config.model,
            query=query,
            documents=documents,
            top_n=config.top_k,
            **self._rerank_kwargs(config),
        )
        return response.results

    def _result_index(self, result: Any) -> int:
        return result.index

    def _result_score(self, result: Any) -> float | None:
        return result.relevance_score
