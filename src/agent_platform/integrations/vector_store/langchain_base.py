import asyncio
from abc import abstractmethod
from typing import Any, Generic
from uuid import UUID, uuid4

from langchain_core.documents import Document as LC_Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.vector_store.base import (
    BaseVectorStore,
    ConfigT,
)
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.score import Score


class LangChainVectorStore(BaseVectorStore[ConfigT], Generic[ConfigT]):
    def __init__(self, embeddings: Embeddings | None = None) -> None:
        self._embeddings = embeddings

    @abstractmethod
    def _default_config(self) -> ConfigT: ...

    @abstractmethod
    def _build_client(self, config: ConfigT) -> VectorStore: ...

    def _search_kwargs(
        self, config: ConfigT, filter: dict[str, Any] | None
    ) -> dict[str, Any]:
        return {"filter": filter}

    async def add(
        self, documents: list[TextChunk], config: ConfigT | None = None
    ) -> None:
        config = config or self._default_config()
        client = self._build_client(config)
        lc_docs = [_chunk_to_lc(doc) for doc in documents]
        await client.aadd_documents(lc_docs)

    async def delete(
        self, document_ids: list[UUID], config: ConfigT | None = None
    ) -> None:
        config = config or self._default_config()
        client = self._build_client(config)
        ids = [str(doc_id) for doc_id in document_ids]
        await asyncio.to_thread(client.delete, ids)

    @error_logged(re_raise=ProviderError, message="Vector store search failed")
    @with_retry()
    async def search(
        self,
        query_vector: list[float],
        k: int = 5,
        config: ConfigT | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        config = config or self._default_config()
        client = self._build_client(config)
        results = await client.asimilarity_search_by_vector(
            query_vector, k, **self._search_kwargs(config, filter)
        )
        return [_lc_to_chunk(c) for c in results]

    @error_logged(re_raise=ProviderError, message="Vector store search failed")
    @with_retry()
    async def search_with_scores(
        self,
        query_vector: list[float],
        k: int = 5,
        config: ConfigT | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[TextChunk, Score]]:
        config = config or self._default_config()
        client = self._build_client(config)
        results = await client.asimilarity_search_with_score(
            query_vector, k, **self._search_kwargs(config, filter)
        )
        return [
            (_lc_to_chunk(doc), Score.similarity(float(score)))
            for doc, score in results
        ]


# --- Mappers ---


def _chunk_to_lc(chunk: TextChunk) -> LC_Document:
    metadata = chunk.metadata or {}
    return LC_Document(
        page_content=chunk.text,
        metadata={
            "document_id": str(chunk.document_id) if chunk.document_id else None,
            "index": chunk.index,
            "chunk_id": str(chunk.id),
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "format": chunk.format.value if chunk.format else None,
            "source": metadata.get("source"),
            "language": metadata.get("language"),
            "extra": metadata.get("extra"),
        },
    )


def _lc_to_chunk(lc_chunk: LC_Document) -> TextChunk:
    m = lc_chunk.metadata
    text = lc_chunk.page_content
    from agent_platform.core.schemas.enums import Language

    return TextChunk(
        id=UUID(m["chunk_id"]) if m.get("chunk_id") else uuid4(),
        document_id=UUID(m["document_id"]) if m.get("document_id") else None,
        text=text,
        index=m.get("index") or 0,
        start_char=m.get("start_index"),
        end_char=(m.get("start_index") or 0) + len(text)
        if m.get("start_index")
        else None,
        metadata={
            "source": m.get("source"),
            "language": Language(m["language"]) if m.get("language") else None,
            "extra": m.get("extra") or {},
        },
    )
