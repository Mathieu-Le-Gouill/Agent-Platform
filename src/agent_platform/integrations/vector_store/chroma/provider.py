from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

import chromadb
import chromadb.api

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.credentials import ChromaCredentials
from agent_platform.integrations.vector_store.chroma.config import ChromaConfig
from agent_platform.integrations.vector_store.chroma.mappers import (
    chunk_to_metadata,
    row_to_chunk,
)


class ChromaStore(BaseVectorStore[ChromaConfig]):
    def __init__(self, credentials: ChromaCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, ChromaCredentials)

    def _default_config(self) -> ChromaConfig:
        return ChromaConfig()

    async def _client(self, config: ChromaConfig) -> Any:
        if self._credentials.api_key:
            return await asyncio.to_thread(
                chromadb.CloudClient,
                tenant=config.tenant,
                database=config.database,
                api_key=self._credentials.api_key.get_secret_value(),
            )
        if config.persist_directory is not None:
            return await asyncio.to_thread(
                chromadb.PersistentClient,
                path=config.persist_directory,
                tenant=config.tenant,
                database=config.database,
            )
        return await chromadb.AsyncHttpClient(
            host=config.host,
            port=config.port,
            ssl=config.ssl,
            tenant=config.tenant,
            database=config.database,
        )

    def _is_async(self, client: Any) -> bool:
        return isinstance(client, chromadb.api.AsyncClientAPI)

    async def _call(
        self, is_async: bool, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if is_async:
            result: Awaitable[Any] = func(*args, **kwargs)
            return await result
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _collection(self, client: Any, config: ChromaConfig) -> tuple[Any, bool]:
        is_async = self._is_async(client)
        collection = await self._call(
            is_async, client.get_or_create_collection, name=config.collection_name
        )
        return collection, is_async

    def _where(self, filter: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filter:
            return None
        if len(filter) == 1:
            return dict(filter)
        return {"$and": [{key: value} for key, value in filter.items()]}

    async def add(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: ChromaConfig | None = None,
    ) -> None:
        config = config or self._default_config()
        if not documents:
            return
        client = await self._client(config)
        collection, is_async = await self._collection(client, config)
        await self._call(
            is_async,
            collection.add,
            ids=[str(doc.id) for doc in documents],
            embeddings=vectors,
            documents=[doc.text for doc in documents],
            metadatas=[chunk_to_metadata(doc) for doc in documents],
        )

    async def delete(
        self, document_ids: list[UUID], config: ChromaConfig | None = None
    ) -> None:
        config = config or self._default_config()
        if not document_ids:
            return
        client = await self._client(config)
        collection, is_async = await self._collection(client, config)
        await self._call(
            is_async,
            collection.delete,
            ids=[str(doc_id) for doc_id in document_ids],
        )

    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        config: ChromaConfig | None = None,
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
        config: ChromaConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
        client = await self._client(config)
        collection, is_async = await self._collection(client, config)
        result = await self._call(
            is_async,
            collection.query,
            query_embeddings=[query_vector],
            n_results=k,
            where=self._where(filter),
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            (
                row_to_chunk(id_, text, metadata),
                # Assumes a cosine-configured collection (`hnsw:space=cosine`),
                # the common setup for text embeddings; the collection is
                # pre-existing and not created by this provider, so its
                # actual metric isn't otherwise known here.
                Score.similarity(min(1.0, max(0.0, 1.0 - distance))),
            )
            for id_, text, metadata, distance in zip(
                ids, documents, metadatas, distances
            )
        ]
