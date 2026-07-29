from __future__ import annotations

from typing import Any
from uuid import UUID

import weaviate
import weaviate.auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.client import WeaviateAsyncClient
from weaviate.collections.classes.data import DataObject
from weaviate.collections.classes.filters import _Filters

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.credentials import WeaviateCredentials
from agent_platform.integrations.vector_store.weaviate.config import WeaviateConfig
from agent_platform.integrations.vector_store.weaviate.mappers import (
    chunk_to_properties,
    object_to_chunk,
)


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


class WeaviateStore(BaseVectorStore[WeaviateConfig]):
    def __init__(self, credentials: WeaviateCredentials | None = None) -> None:
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

    async def _connect(self, config: WeaviateConfig) -> WeaviateAsyncClient:
        host, port, secure = self._connection_target(config)
        if self._credentials.api_key:
            client = weaviate.use_async_with_custom(
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
        else:
            client = weaviate.use_async_with_local(
                host=host, port=port, grpc_port=config.grpc_port
            )
        await client.connect()
        return client

    def _collection(self, client: WeaviateAsyncClient, config: WeaviateConfig) -> Any:
        collection = client.collections.get(config.collection_name)
        if config.namespace:
            collection = collection.with_tenant(config.namespace)
        return collection

    def _filter(self, filter: dict[str, Any] | None) -> _Filters | None:
        if not filter:
            return None
        conditions = [
            Filter.by_property(key).equal(value) for key, value in filter.items()
        ]
        return conditions[0] if len(conditions) == 1 else Filter.all_of(conditions)

    async def add(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: WeaviateConfig | None = None,
    ) -> None:
        config = config or self._default_config()
        client = await self._connect(config)
        try:
            collection = self._collection(client, config)
            objects = [
                DataObject(
                    properties=chunk_to_properties(doc, config.text_key),
                    uuid=doc.id,
                    vector=vector,
                )
                for doc, vector in zip(documents, vectors)
            ]
            if objects:
                await collection.data.insert_many(objects)
        finally:
            await client.close()

    async def delete(
        self, document_ids: list[UUID], config: WeaviateConfig | None = None
    ) -> None:
        config = config or self._default_config()
        client = await self._connect(config)
        try:
            collection = self._collection(client, config)
            if document_ids:
                await collection.data.delete_many(
                    where=Filter.by_id().contains_any(
                        [str(doc_id) for doc_id in document_ids]
                    )
                )
        finally:
            await client.close()

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
        client = await self._connect(config)
        try:
            collection = self._collection(client, config)
            result = await collection.query.near_vector(
                near_vector=query_vector,
                limit=k,
                filters=self._filter(filter),
            )
            return [object_to_chunk(obj, config.text_key) for obj in result.objects]
        finally:
            await client.close()

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
        client = await self._connect(config)
        try:
            collection = self._collection(client, config)
            result = await collection.query.near_vector(
                near_vector=query_vector,
                limit=k,
                filters=self._filter(filter),
                return_metadata=MetadataQuery(distance=True),
            )
            return [
                (
                    object_to_chunk(obj, config.text_key),
                    Score.similarity(
                        min(1.0, max(0.0, 1.0 - (obj.metadata.distance or 0.0)))
                    ),
                )
                for obj in result.objects
            ]
        finally:
            await client.close()
