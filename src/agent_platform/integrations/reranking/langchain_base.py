from abc import abstractmethod
from typing import Generic, Sequence
from uuid import UUID

from langchain_core.documents import (
    BaseDocumentCompressor,
    Document as LC_Document,
)

from agent_platform.core.interfaces.reranking.base import (
    BaseReranker,
    RerankerConfigT,
)
from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.reranking.config import RerankerConfig
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language


class LangChainReranker(
    BaseReranker[TextChunk, RerankerConfigT],
    Generic[RerankerConfigT],
):
    @abstractmethod
    def _client(self, config: RerankerConfigT) -> BaseDocumentCompressor: ...

    @abstractmethod
    def _default_config(self) -> RerankerConfigT: ...

    @error_logged(re_raise=ProviderError, message="Reranking failed")
    @with_retry()
    async def rerank(
        self,
        query: str,
        items: Sequence[TextChunk],
        config: RerankerConfigT | None = None,
    ) -> Sequence[TextChunk]:
        config = config or self._default_config()
        client = self._client(config)
        documents = [_chunk_to_lc(item) for item in items]
        result = await client.acompress_documents(documents, query)

        if config.top_k is not None:
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
        result.append(
            TextChunk(
                id=UUID(m["chunk_id"]) if m.get("chunk_id") else UUID(int=0),
                document_id=UUID(m["document_id"]) if m.get("document_id") else None,
                text=c.page_content,
                index=m.get("index") or 0,
                start_char=m.get("start_index"),
                end_char=(m.get("start_index") or 0) + len(c.page_content)
                if m.get("start_index")
                else None,
                metadata={
                    "source": m.get("source"),
                    "language": Language(m["language"]) if m.get("language") else None,
                    "extra": m.get("extra") or {},
                },
            )
        )
    return result
