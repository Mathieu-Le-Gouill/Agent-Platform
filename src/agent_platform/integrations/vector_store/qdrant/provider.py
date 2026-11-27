from __future__ import annotations

import functools
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse

from agent_platform.core.credentials import (
    ClientOptions,
    resolve_client_options,
    resolve_credentials,
    resolve_timeout,
)
from agent_platform.core.errors import ProviderError, error_logged
from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.core.schemas.vector import SparseVector
from agent_platform.integrations.credentials import QdrantCredentials
from agent_platform.integrations.sdk_errors import translate_sdk_error
from agent_platform.integrations.vector_store._base import reject_namespace
from agent_platform.integrations.vector_store.qdrant.config import QdrantConfig
from agent_platform.integrations.vector_store.qdrant.mappers import (
    chunk_to_payload,
    point_to_chunk,
)

_STORE = "Qdrant"
_SEARCH_ERROR = "Vector store search failed"
_HYBRID_SEARCH_ERROR = "Vector store hybrid search failed"

# The client raises this, with no status, for a rate limit. It is retried, but
# the wait the server asked for (`retry_after_s`) is not honoured: the shared
# classifier reads `Retry-After` only from `exc.response.headers`.
_TRANSIENT_TYPES = (ResourceExhaustedResponse,)
_translate_search = functools.partial(
    translate_sdk_error, operation=_SEARCH_ERROR, transient_types=_TRANSIENT_TYPES
)
_translate_hybrid_search = functools.partial(
    translate_sdk_error,
    operation=_HYBRID_SEARCH_ERROR,
    transient_types=_TRANSIENT_TYPES,
)
# Writes are retried because they are idempotent: an upsert by point id and a
# delete by id leave the same state however often they run.
_ADD_ERROR = "Vector store add failed"
_HYBRID_ADD_ERROR = "Vector store hybrid add failed"
_DELETE_ERROR = "Vector store delete failed"
_translate_add = functools.partial(
    translate_sdk_error, operation=_ADD_ERROR, transient_types=_TRANSIENT_TYPES
)
_translate_hybrid_add = functools.partial(
    translate_sdk_error, operation=_HYBRID_ADD_ERROR, transient_types=_TRANSIENT_TYPES
)
_translate_delete = functools.partial(
    translate_sdk_error, operation=_DELETE_ERROR, transient_types=_TRANSIENT_TYPES
)


class QdrantVectorStoreProvider(BaseVectorStore[QdrantConfig]):
    def __init__(
        self,
        credentials: QdrantCredentials | None = None,
        client_options: ClientOptions | None = None,
    ) -> None:
        self._credentials = resolve_credentials(credentials, QdrantCredentials)
        self._client_options = resolve_client_options(client_options)

    def _default_config(self) -> QdrantConfig:
        return QdrantConfig()

    def _client(self, config: QdrantConfig) -> AsyncQdrantClient:
        timeout = resolve_timeout(config.timeout, self._client_options)
        # qdrant-client makes no retry of its own on a search, so the platform's
        # `with_retry` is the only retry layer.
        return AsyncQdrantClient(
            url=config.url,
            api_key=self._credentials.api_key.get_secret_value()
            if self._credentials.api_key
            else None,
            prefer_grpc=config.prefer_grpc,
            timeout=int(timeout) if timeout is not None else None,
        )

    def _filter(self, filter: dict[str, Any] | None) -> models.Filter | None:
        if not filter:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in filter.items()
            ]
        )

    @error_logged(re_raise=ProviderError, message=_ADD_ERROR)
    @with_retry(translate=_translate_add)
    async def add(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: QdrantConfig | None = None,
    ) -> None:
        config = config or self._default_config()
        reject_namespace(config, _STORE)
        client = self._client(config)
        points = [
            models.PointStruct(
                id=str(doc.id), vector=vector, payload=chunk_to_payload(doc)
            )
            for doc, vector in zip(documents, vectors, strict=True)
        ]
        if points:
            await client.upsert(collection_name=config.collection_name, points=points)

    @error_logged(re_raise=ProviderError, message=_DELETE_ERROR)
    @with_retry(translate=_translate_delete)
    async def delete(
        self, document_ids: list[UUID], config: QdrantConfig | None = None
    ) -> None:
        config = config or self._default_config()
        reject_namespace(config, _STORE)
        client = self._client(config)
        if document_ids:
            await client.delete(
                collection_name=config.collection_name,
                points_selector=models.PointIdsList(
                    points=[str(doc_id) for doc_id in document_ids]
                ),
            )

    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        config: QdrantConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        results = await self.search_with_scores(query_vector, k, config, filter)
        return [chunk for chunk, _ in results]

    @error_logged(re_raise=ProviderError, message=_SEARCH_ERROR)
    @with_retry(translate=_translate_search)
    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
        config: QdrantConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
        reject_namespace(config, _STORE)
        client = self._client(config)
        response = await client.query_points(
            collection_name=config.collection_name,
            query=query_vector,
            limit=k,
            query_filter=self._filter(filter),
        )
        return [
            (
                point_to_chunk(point),
                Score.similarity(min(1.0, max(0.0, point.score))),
            )
            for point in response.points
        ]

    @error_logged(re_raise=ProviderError, message=_HYBRID_ADD_ERROR)
    @with_retry(translate=_translate_hybrid_add)
    async def add_hybrid(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
        config: QdrantConfig | None = None,
    ) -> None:
        config = config or self._default_config()
        reject_namespace(config, _STORE)
        client = self._client(config)
        points = [
            models.PointStruct(
                id=str(doc.id),
                vector={
                    "dense": vector,
                    "sparse": models.SparseVector(
                        indices=sparse.indices, values=sparse.values
                    ),
                },
                payload=chunk_to_payload(doc),
            )
            for doc, vector, sparse in zip(
                documents, vectors, sparse_vectors, strict=True
            )
        ]
        if points:
            await client.upsert(collection_name=config.collection_name, points=points)

    @error_logged(re_raise=ProviderError, message=_HYBRID_SEARCH_ERROR)
    @with_retry(translate=_translate_hybrid_search)
    async def search_hybrid(
        self,
        query_vector: list[float],
        sparse_vector: SparseVector,
        k: int = 5,
        config: QdrantConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
        reject_namespace(config, _STORE)
        client = self._client(config)
        response = await client.query_points(
            collection_name=config.collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=k,
                    filter=self._filter(filter),
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices, values=sparse_vector.values
                    ),
                    using="sparse",
                    limit=k,
                    filter=self._filter(filter),
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
        )
        return [
            (
                point_to_chunk(point),
                Score.similarity(min(1.0, max(0.0, point.score))),
            )
            for point in response.points
        ]
