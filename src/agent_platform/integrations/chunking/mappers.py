from __future__ import annotations

from uuid import UUID, uuid4

from langchain_core.documents import Document as LC_Document

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.document import TextDocument
from agent_platform.core.schemas.enums import DocumentFormat, Language

__all__ = ["doc_to_lc", "lc_to_chunks"]


def doc_to_lc(doc: TextDocument) -> LC_Document:
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


def lc_to_chunks(lc_chunks: list[LC_Document]) -> list[TextChunk]:
    result = []
    for c in lc_chunks:
        m = c.metadata
        chunk_id = m.get("chunk_id")
        document_id = m.get("document_id")
        language = m.get("language")
        start_char = m.get("start_index") or m.get("start_char")
        format = m.get("format")

        try:
            parsed_chunk_id = UUID(chunk_id) if chunk_id else None
        except ValueError:
            parsed_chunk_id = None

        try:
            parsed_document_id = UUID(document_id) if document_id else None
        except ValueError:
            parsed_document_id = None

        result.append(
            TextChunk(
                id=parsed_chunk_id or uuid4(),
                document_id=parsed_document_id,
                text=c.page_content,
                index=m.get("index", 0),
                format=DocumentFormat(format) if format else DocumentFormat.UNKNOWN,
                start_char=start_char,
                end_char=(start_char + len(c.page_content))
                if start_char is not None
                else None,
                metadata={
                    "source": m.get("source"),
                    "language": Language(language) if language else None,
                    "extra": m.get("extra", {}),
                },
            )
        )
    return result
