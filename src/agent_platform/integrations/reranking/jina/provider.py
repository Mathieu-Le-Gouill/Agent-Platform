from __future__ import annotations

from typing import Any

import httpx

from agent_platform.core.credentials import resolve_credentials
from agent_platform.integrations.credentials import JinaCredentials
from agent_platform.integrations.reranking._base import NativeReranker
from agent_platform.integrations.reranking.jina.config import JinaRerankerConfig

_API_URL = "https://api.jina.ai/v1/rerank"


class JinaRerankerProvider(
    NativeReranker[JinaRerankerConfig, httpx.AsyncClient, httpx.Client, Any]
):
    def __init__(self, credentials: JinaCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, JinaCredentials)

    def _default_config(self) -> JinaRerankerConfig:
        return JinaRerankerConfig()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._credentials.api_key is not None:
            headers["Authorization"] = (
                f"Bearer {self._credentials.api_key.get_secret_value()}"
            )
        return headers

    def _async_client(self, config: JinaRerankerConfig) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=self._headers())

    def _sync_client(self, config: JinaRerankerConfig) -> httpx.Client:
        return httpx.Client(headers=self._headers())

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
                _API_URL, json=self._payload(query, documents, config)
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
                _API_URL, json=self._payload(query, documents, config)
            )
            response.raise_for_status()
            return response.json()["results"]

    def _result_index(self, result: Any) -> int:
        return result["index"]

    def _result_score(self, result: Any) -> float | None:
        return result.get("relevance_score")
