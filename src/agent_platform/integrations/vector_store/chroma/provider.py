from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

import chromadb
import chromadb.api

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.credentials import ChromaCredentials
from agent_platform.integrations.vector_store.chroma.config import ChromaConfig


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
            metadatas=[_chunk_to_metadata(doc) for doc in documents],
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
                _row_to_chunk(id_, text, metadata),
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


# --- Mappers ---


def _chunk_to_metadata(chunk: TextChunk) -> dict[str, Any]:
    metadata = chunk.metadata or {}
    result: dict[str, Any] = {"extra": json.dumps(metadata.get("extra") or {})}
    if chunk.document_id:
        result["document_id"] = str(chunk.document_id)
    if chunk.index is not None:
        result["index"] = chunk.index
    if chunk.start_char is not None:
        result["start_char"] = chunk.start_char
    if chunk.end_char is not None:
        result["end_char"] = chunk.end_char
    if chunk.format:
        result["format"] = chunk.format.value
    if metadata.get("source") is not None:
        result["source"] = metadata["source"]
    if metadata.get("language") is not None:
        result["language"] = metadata["language"]
    return result


def _row_to_chunk(
    id_: str, text: str | None, metadata: dict[str, Any] | None
) -> TextChunk:
    metadata = metadata or {}
    extra: dict[str, Any] = {}
    if metadata.get("extra"):
        try:
            extra = json.loads(metadata["extra"])
        except (TypeError, ValueError):
            extra = {}
    return TextChunk(
        id=UUID(id_),
        document_id=UUID(metadata["document_id"])
        if metadata.get("document_id")
        else None,
        text=text or "",
        index=metadata.get("index") or 0,
        start_char=metadata.get("start_char"),
        end_char=metadata.get("end_char"),
        metadata={
            "source": metadata.get("source"),
            "language": Language(metadata["language"])
            if metadata.get("language")
            else None,
            "extra": extra,
        },
    )
