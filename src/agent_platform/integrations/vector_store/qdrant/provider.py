from __future__ import annotations

from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.core.schemas.vector import SparseVector
from agent_platform.integrations.credentials import QdrantCredentials
from agent_platform.integrations.vector_store.qdrant.config import QdrantConfig
from agent_platform.integrations.vector_store.qdrant.mappers import (
    chunk_to_payload,
    point_to_chunk,
)


class QdrantVectorStoreProvider(BaseVectorStore[QdrantConfig]):
    def __init__(self, credentials: QdrantCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, QdrantCredentials)

    def _default_config(self) -> QdrantConfig:
        return QdrantConfig()

    def _client(self, config: QdrantConfig) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            url=config.url,
            api_key=self._credentials.api_key.get_secret_value()
            if self._credentials.api_key
            else None,
            prefer_grpc=config.prefer_grpc,
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

    async def add(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: QdrantConfig | None = None,
    ) -> None:
        config = config or self._default_config()
        client = self._client(config)
        points = [
            models.PointStruct(
                id=str(doc.id), vector=vector, payload=chunk_to_payload(doc)
            )
            for doc, vector in zip(documents, vectors)
        ]
        if points:
            await client.upsert(collection_name=config.collection_name, points=points)

    async def delete(
        self, document_ids: list[UUID], config: QdrantConfig | None = None
    ) -> None:
        config = config or self._default_config()
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

    @error_logged(re_raise=ProviderError, message="Vector store search failed")
    @with_retry()
    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
        config: QdrantConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
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

    async def add_hybrid(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
        config: QdrantConfig | None = None,
    ) -> None:
        config = config or self._default_config()
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
            for doc, vector, sparse in zip(documents, vectors, sparse_vectors)
        ]
        if points:
            await client.upsert(collection_name=config.collection_name, points=points)

    @error_logged(re_raise=ProviderError, message="Vector store hybrid search failed")
    @with_retry()
    async def search_hybrid(
        self,
        query_vector: list[float],
        sparse_vector: SparseVector,
        k: int = 5,
        config: QdrantConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
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
