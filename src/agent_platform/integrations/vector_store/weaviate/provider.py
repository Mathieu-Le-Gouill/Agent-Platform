from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import UUID

import weaviate
import weaviate.auth
from langchain_core.embeddings import Embeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import Filter

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.credentials import WeaviateCredentials
from agent_platform.integrations.vector_store.langchain_base import (
    LangChainVectorStore,
    _chunk_to_lc,
    _lc_to_chunk,
)
from agent_platform.integrations.vector_store.weaviate.config import WeaviateConfig


def _parse_url(url: str) -> tuple[str, int, bool]:
    secure = url.startswith("https")
    remainder = url.split("://", 1)[-1]
    if ":" in remainder:
        host, port_str = remainder.rsplit(":", 1)
        port = int(port_str.split("/")[0])
    else:
        host = remainder.split("/")[0]
        port = 443 if secure else 80
    return host, port, secure


class WeaviateStore(LangChainVectorStore[WeaviateConfig]):
    def __init__(
        self,
        credentials: WeaviateCredentials | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        super().__init__(embeddings)
        self._credentials = resolve_credentials(credentials, WeaviateCredentials)

    def _default_config(self) -> WeaviateConfig:
        return WeaviateConfig()

    def _connection_target(self, config: WeaviateConfig) -> tuple[str, int, bool]:
        host, port, secure = _parse_url(self._credentials.url)
        if config.http_host is not None:
            host = config.http_host
        if config.http_port is not None:
            port = config.http_port
        return host, port, secure

    def _connect(self, config: WeaviateConfig) -> weaviate.WeaviateClient:
        host, port, secure = self._connection_target(config)
        if self._credentials.api_key:
            return weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=secure,
                grpc_host=host,
                grpc_port=config.grpc_port,
                grpc_secure=secure,
                auth_credentials=weaviate.auth.AuthApiKey(
                    self._credentials.api_key.get_secret_value()
                ),
            )
        return weaviate.connect_to_local(
            host=host, port=port, grpc_port=config.grpc_port
        )

    def _build_client(
        self,
        config: WeaviateConfig,
        raw_client: weaviate.WeaviateClient | None = None,
    ) -> WeaviateVectorStore:
        raw_client = raw_client if raw_client is not None else self._connect(config)
        return WeaviateVectorStore(
            client=raw_client,
            index_name=config.collection_name,
            text_key=config.text_key,
            embedding=self._embeddings,
            use_multi_tenancy=bool(config.namespace),
        )

    def _search_kwargs(
        self, config: WeaviateConfig, filter: dict[str, Any] | None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if filter:
            conditions = [
                Filter.by_property(key).equal(value) for key, value in filter.items()
            ]
            kwargs["filters"] = (
                conditions[0] if len(conditions) == 1 else Filter.all_of(conditions)
            )
        if config.namespace:
            kwargs["tenant"] = config.namespace
        return kwargs

    async def add(
        self, documents: list[TextChunk], config: WeaviateConfig | None = None
    ) -> None:
        config = config or self._default_config()
        raw_client = self._connect(config)
        try:
            client = self._build_client(config, raw_client=raw_client)
            lc_docs = [_chunk_to_lc(doc) for doc in documents]
            await client.aadd_documents(lc_docs)
        finally:
            raw_client.close()

    async def delete(
        self, document_ids: list[UUID], config: WeaviateConfig | None = None
    ) -> None:
        config = config or self._default_config()
        raw_client = self._connect(config)
        try:
            client = self._build_client(config, raw_client=raw_client)
            ids = [str(doc_id) for doc_id in document_ids]
            await asyncio.to_thread(client.delete, ids)
        finally:
            raw_client.close()

    @error_logged(re_raise=ProviderError, message="Vector store search failed")
    @with_retry()
    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        config: WeaviateConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        config = config or self._default_config()
        raw_client = self._connect(config)
        try:
            client = self._build_client(config, raw_client=raw_client)
            results = await client.asimilarity_search_by_vector(
                query_vector, k, **self._search_kwargs(config, filter)
            )
            return [_lc_to_chunk(c) for c in results]
        finally:
            raw_client.close()

    @error_logged(re_raise=ProviderError, message="Vector store search failed")
    @with_retry()
    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
        config: WeaviateConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
        raw_client = self._connect(config)
        try:
            client = self._build_client(config, raw_client=raw_client)
            # `WeaviateVectorStore` has no public by-vector search that also
            # returns scores; `return_score` is forwarded to `_perform_asearch`
            # (undocumented but supported), giving `list[tuple[Document, float]]`
            # instead of the `list[Document]` the stub declares.
            results = cast(
                list[tuple[Any, float]],
                await client.asimilarity_search_by_vector(
                    query_vector,
                    k,
                    return_score=True,
                    **self._search_kwargs(config, filter),
                ),
            )
            return [
                (_lc_to_chunk(doc), Score.similarity(float(score)))
                for doc, score in results
            ]
        finally:
            raw_client.close()
