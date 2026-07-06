from abc import abstractmethod
from typing import Sequence
from uuid import UUID

from langchain_core.documents import (
    BaseDocumentCompressor,
    Document as LC_Document,
)

from agent_platform.integrations.reranking.base import BaseReranker
from agent_platform.integrations.reranking.config import RerankerConfig
from agent_platform.models.chunk import TextChunk
from agent_platform.models.enums import Language


class LangChainReranker(BaseReranker[TextChunk, RerankerConfig]):

    @abstractmethod
    def _client(self) -> BaseDocumentCompressor:
        ...

    async def rerank(
        self,
        query: str,
        items: Sequence[TextChunk],
        config: RerankerConfig = RerankerConfig(),
    ) -> Sequence[TextChunk]:
        client = self._client()
        documents = [_chunk_to_lc(item) for item in items]
        result = await client.acompress_documents(documents, query)

        if config is not None and config.top_k is not None:
            result = result[: config.top_k]

        return _lc_to_chunks(result)


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


def _lc_to_chunks(lc_chunks: Sequence[LC_Document]) -> list[TextChunk]:
    result = []
    for c in lc_chunks:
        m = c.metadata
        result.append(TextChunk(
            id=UUID(m["chunk_id"]) if m.get("chunk_id") else UUID(int=0),
            document_id=UUID(m["document_id"]) if m.get("document_id") else None,
            text=c.page_content,
            index=m.get("index") or 0,
            start_char=m.get("start_index"),
            end_char=(m.get("start_index") or 0) + len(c.page_content) if m.get("start_index") else None,
            metadata={
                "source": m.get("source"),
                "language": Language(m["language"]) if m.get("language") else None,
                "extra": m.get("extra") or {},
            },
        ))
    return result
