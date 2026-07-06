from abc import abstractmethod
import asyncio
from uuid import UUID

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document as LC_Document

from agent_platform.integrations.vector_store.base import BaseVectorStore
from agent_platform.models.chunk import TextChunk
from agent_platform.models.score import Score
from uuid import uuid4


class LangChainVectorStore(BaseVectorStore):
    _client: VectorStore

    def __init__(self):
        self._client = self._build_client()

    @abstractmethod
    def _build_client(self) -> VectorStore: ...

    async def add(self, documents: list[TextChunk]) -> None:
        lc_docs = [_chunk_to_lc(doc) for doc in documents]
        await asyncio.to_thread(self._client.aadd_documents, lc_docs)

    async def search(self, query_vector: list[float], k: int = 5) -> list[TextChunk]:
        results = await asyncio.to_thread(
            self._client.similarity_search_by_vector,
            query_vector,
            k,
        )
        return [_lc_to_chunk(c) for c in results]

    async def search_with_scores(
        self, query_vector: list[float], k: int = 5
    ) -> list[tuple[TextChunk, Score]]:
        results = await asyncio.to_thread(
            self._client.similarity_search_with_score,
            query_vector,
            k,
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
    from agent_platform.models.enums import Language

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
