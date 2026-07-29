from __future__ import annotations

from typing import Any
from uuid import UUID

from pinecone import PineconeAsyncio

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import (
    ProviderError,
    error_logged,
    require_secret,
    with_retry,
)
from agent_platform.core.interfaces.vector_store.base import BaseVectorStore
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.credentials import PineconeCredentials
from agent_platform.integrations.vector_store.pinecone.config import PineconeConfig
from agent_platform.integrations.vector_store.pinecone.mappers import (
    chunk_to_metadata,
    match_to_chunk,
)


class PineconeStore(BaseVectorStore[PineconeConfig]):
    def __init__(self, credentials: PineconeCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, PineconeCredentials)

    def _default_config(self) -> PineconeConfig:
        return PineconeConfig()

    def _api_key(self) -> str:
        return require_secret(
            self._credentials.api_key, "Pinecone API key is required"
        ).get_secret_value()

    async def _resolve_host(self, config: PineconeConfig) -> str:
        if config.host is not None:
            return config.host
        async with PineconeAsyncio(api_key=self._api_key()) as pc:
            index_model = await pc.describe_index(config.collection_name)
            return index_model.host

    async def _index(self, config: PineconeConfig) -> Any:
        host = await self._resolve_host(config)
        pc = PineconeAsyncio(api_key=self._api_key())
        return pc.IndexAsyncio(host=host)

    async def add(
        self,
        documents: list[TextChunk],
        vectors: list[list[float]],
        config: PineconeConfig | None = None,
    ) -> None:
        config = config or self._default_config()
        if not documents:
            return
        async with await self._index(config) as index:
            payload = [
                {
                    "id": str(doc.id),
                    "values": vector,
                    "metadata": chunk_to_metadata(doc),
                }
                for doc, vector in zip(documents, vectors)
            ]
            await index.upsert(vectors=payload, namespace=config.namespace)

    async def delete(
        self, document_ids: list[UUID], config: PineconeConfig | None = None
    ) -> None:
        config = config or self._default_config()
        if not document_ids:
            return
        async with await self._index(config) as index:
            await index.delete(
                ids=[str(doc_id) for doc_id in document_ids],
                namespace=config.namespace,
            )

    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        config: PineconeConfig | None = None,
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
        config: PineconeConfig | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
        async with await self._index(config) as index:
            response = await index.query(
                vector=query_vector,
                top_k=k,
                namespace=config.namespace,
                filter=filter or None,
                include_metadata=True,
            )
        return [
            (
                match_to_chunk(match),
                Score.similarity(min(1.0, max(0.0, match.score))),
            )
            for match in response.matches
        ]
