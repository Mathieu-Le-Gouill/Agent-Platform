from __future__ import annotations

import json
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
from agent_platform.core.schemas.enums import Language
from agent_platform.core.schemas.score import Score
from agent_platform.integrations.credentials import PineconeCredentials
from agent_platform.integrations.vector_store.pinecone.config import PineconeConfig


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
                    "metadata": _chunk_to_metadata(doc),
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
                _match_to_chunk(match),
                Score.similarity(min(1.0, max(0.0, match.score))),
            )
            for match in response.matches
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
    result["text"] = chunk.text
    if metadata.get("source") is not None:
        result["source"] = metadata["source"]
    if metadata.get("language") is not None:
        result["language"] = metadata["language"]
    return result


def _match_to_chunk(match: Any) -> TextChunk:
    metadata = match.metadata or {}
    extra: dict[str, Any] = {}
    if metadata.get("extra"):
        try:
            extra = json.loads(metadata["extra"])
        except (TypeError, ValueError):
            extra = {}
    return TextChunk(
        id=UUID(str(match.id)),
        document_id=UUID(metadata["document_id"])
        if metadata.get("document_id")
        else None,
        text=metadata.get("text") or "",
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
