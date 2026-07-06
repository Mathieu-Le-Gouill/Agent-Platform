from abc import abstractmethod
from typing import Sequence

from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document as LC_Document

from agent_platform.models.chunk import TextChunk
from agent_platform.models.document import TextDocument
from agent_platform.models.enums import Language, DocumentFormat

from agent_platform.integrations.chunking.base import BaseChunker
from agent_platform.integrations.chunking.config import ChunkerConfig


class LangChainChunker(BaseChunker[TextDocument, TextChunk, ChunkerConfig]):

    @abstractmethod
    def _splitter(self, config: ChunkerConfig | None) -> TextSplitter:
        ...
        

    async def chunk(
        self,
        documents: Sequence[TextDocument],
        config: ChunkerConfig | None = None,
    ) -> list[TextChunk]:
        splitter = self._splitter(config)
        lc_documents = [_doc_to_lc(doc) for doc in documents]
        lc_chunks = splitter.split_documents(lc_documents)
        return _lc_to_chunks(lc_chunks)


# --- Mappers ---

def _doc_to_lc(doc: TextDocument) -> LC_Document:
    return LC_Document(
        page_content=doc.text,
        metadata={
            "document_id": str(doc.id),
            "source": doc.source,
            "title": doc.metadata.title,
            "author": doc.metadata.author,
            "description": doc.metadata.description,
            "format": doc.format.value if doc.format else None,
            "created_at": doc.metadata.created_at,
            "modified_at": doc.metadata.modified_at,
            "extra": doc.metadata.extra,
            "encoding": doc.encoding,
            "language": doc.language.value if doc.language else None,
        },
    )


def _lc_to_chunks(lc_chunks: list[LC_Document]) -> list[TextChunk]:
    from uuid import UUID
    result = []
    for c in lc_chunks:
        m = c.metadata
        chunk_id = m.get("chunk_id")
        document_id = m.get("document_id")
        language = m.get("language")
        start_char = m.get("start_index") or m.get("start_char")
        format = m.get("format")

        result.append(TextChunk(
            id=UUID(chunk_id) if chunk_id else __import__("uuid").uuid4(),
            document_id=UUID(document_id) if document_id else None,
            text=c.page_content,
            index=m.get("index") or 0,
            format=DocumentFormat(format),
            start_char=start_char,
            end_char=(start_char + len(c.page_content)) if start_char is not None else None,
            metadata={
                "source": m.get("source"),
                "language": Language(language) if language else None,
                "extra": m.get("extra") or {},
            },
        ))
    return result