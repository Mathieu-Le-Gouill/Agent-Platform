from __future__ import annotations

from typing import Any

import httpx

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_timeout,
)
from agent_platform.integrations.credentials import JinaCredentials
from agent_platform.integrations.reranking._base import NativeReranker
from agent_platform.integrations.reranking.jina.config import JinaRerankerConfig

_DEFAULT_BASE_URL = "https://api.jina.ai"
_RERANK_PATH = "/v1/rerank"


class JinaRerankerProvider(
    NativeReranker[JinaRerankerConfig, httpx.AsyncClient, httpx.Client, Any]
):
    def __init__(
        self,
        credentials: JinaCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, JinaCredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> JinaRerankerConfig:
        return JinaRerankerConfig()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._credentials.api_key is not None:
            headers["Authorization"] = (
                f"Bearer {self._credentials.api_key.get_secret_value()}"
            )
        return headers

    def _client_kwargs(self, config: JinaRerankerConfig) -> dict[str, Any]:
        # httpx has no client-level retry-count knob (unlike the openai/cohere
        # SDKs), so `max_retries` isn't wired here; the platform's own
        # `@with_retry()` wrapper on `arerank` already covers that.
        kwargs: dict[str, Any] = {
            "headers": self._headers(),
            "base_url": self._client_options.base_url or _DEFAULT_BASE_URL,
        }
        timeout = resolve_timeout(config.timeout, self._client_options)
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _async_client(self, config: JinaRerankerConfig) -> httpx.AsyncClient:
        return httpx.AsyncClient(**self._client_kwargs(config))

    def _sync_client(self, config: JinaRerankerConfig) -> httpx.Client:
        return httpx.Client(**self._client_kwargs(config))

    def _payload(
        self, query: str, documents: list[str], config: JinaRerankerConfig
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        if config.top_k is not None:
            # `top_n` defaults to returning every document server-side, so
            # only set it when the caller actually wants a server-side cap.
            payload["top_n"] = config.top_k
        return payload

    def _invoke_sync(
        self,
        client: httpx.Client,
        query: str,
        documents: list[str],
        config: JinaRerankerConfig,
    ) -> Any:
        with client:
            response = client.post(
                _RERANK_PATH, json=self._payload(query, documents, config)
            )
            response.raise_for_status()
            return response.json()["results"]

    async def _invoke_async(
        self,
        client: httpx.AsyncClient,
        query: str,
        documents: list[str],
        config: JinaRerankerConfig,
    ) -> Any:
        async with client:
            response = await client.post(
                _RERANK_PATH, json=self._payload(query, documents, config)
            )
            response.raise_for_status()
            return response.json()["results"]

    def _result_index(self, result: Any) -> int:
        return result["index"]

    def _result_score(self, result: Any) -> float | None:
        return result.get("relevance_score")
