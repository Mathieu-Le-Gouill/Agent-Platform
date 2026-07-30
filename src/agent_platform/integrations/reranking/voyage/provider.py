from __future__ import annotations

from typing import Any

import voyageai

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.integrations.credentials import VoyageCredentials
from agent_platform.integrations.reranking._base import NativeReranker
from agent_platform.integrations.reranking.voyage.config import VoyageRerankerConfig


class VoyageRerankerProvider(
    NativeReranker[VoyageRerankerConfig, voyageai.AsyncClient, voyageai.Client, Any]
):
    def __init__(
        self,
        credentials: VoyageCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, VoyageCredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> VoyageRerankerConfig:
        return VoyageRerankerConfig()

    def _client_kwargs(self, config: VoyageRerankerConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._credentials.api_key is not None:
            kwargs["api_key"] = self._credentials.api_key.get_secret_value()
        kwargs["base_url"] = self._client_options.base_url
        # voyageai's own SDK default for max_retries is 0 (not 3); resolve
        # against ClientOptions so behavior stays consistent with the rest
        # of the platform rather than inheriting voyage's unusually low default.
        kwargs["max_retries"] = resolve_max_retries(
            config.max_retries, self._client_options
        )
        timeout = resolve_timeout(config.timeout, self._client_options)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _async_client(self, config: VoyageRerankerConfig) -> voyageai.AsyncClient:
        return voyageai.AsyncClient(**self._client_kwargs(config))

    def _sync_client(self, config: VoyageRerankerConfig) -> voyageai.Client:
        return voyageai.Client(**self._client_kwargs(config))

    def _rerank_kwargs(self, config: VoyageRerankerConfig) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if config.truncation is not None:
            kwargs["truncation"] = config.truncation
        return kwargs

    def _invoke_sync(
        self,
        client: voyageai.Client,
        query: str,
        documents: list[str],
        config: VoyageRerankerConfig,
    ) -> Any:
        response = client.rerank(
            query=query,
            documents=documents,
            model=config.model,
            top_k=config.top_k,
            **self._rerank_kwargs(config),
        )
        return response.results

    async def _invoke_async(
        self,
        client: voyageai.AsyncClient,
        query: str,
        documents: list[str],
        config: VoyageRerankerConfig,
    ) -> Any:
        response = await client.rerank(
            query=query,
            documents=documents,
            model=config.model,
            top_k=config.top_k,
            **self._rerank_kwargs(config),
        )
        return response.results

    def _result_index(self, result: Any) -> int:
        return result.index

    def _result_score(self, result: Any) -> float | None:
        return result.relevance_score
