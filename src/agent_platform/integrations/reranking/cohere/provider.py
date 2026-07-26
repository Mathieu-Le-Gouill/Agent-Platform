from __future__ import annotations

from typing import Any

import cohere

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import require_secret
from agent_platform.integrations.credentials import CohereCredentials
from agent_platform.integrations.reranking._base import NativeReranker
from agent_platform.integrations.reranking.cohere.config import CohereRerankerConfig


class CohereRerankerProvider(
    NativeReranker[CohereRerankerConfig, cohere.AsyncClient, cohere.Client, Any]
):
    def __init__(self, credentials: CohereCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, CohereCredentials)

    def _default_config(self) -> CohereRerankerConfig:
        return CohereRerankerConfig()

    def _api_key(self) -> str:
        api_key = require_secret(
            self._credentials.api_key, "Cohere API key is required"
        )
        return api_key.get_secret_value()

    def _async_client(self, config: CohereRerankerConfig) -> cohere.AsyncClient:
        return cohere.AsyncClient(api_key=self._api_key())

    def _sync_client(self, config: CohereRerankerConfig) -> cohere.Client:
        return cohere.Client(api_key=self._api_key())

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
