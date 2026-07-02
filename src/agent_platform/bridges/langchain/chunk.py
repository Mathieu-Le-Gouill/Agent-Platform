from langchain_core.documents import Document as LCDocument
from uuid import UUID
from typing import Sequence

from agent_platform.models.enums.file_format import DocumentFormat
from agent_platform.utils import parse_uuid
from agent_platform.models.chunk import TextChunk
from agent_platform.models.enums.language import Language


def from_langchain_many(lc_chunks: Sequence[LCDocument]) -> list[TextChunk]:

    counters: dict[UUID, int] = {}

    chunks: list[TextChunk] = []

    for lc_chunk in lc_chunks:

        m = lc_chunk.metadata

        chunk_id = parse_uuid(m.get("chunk_id"))
        document_id = parse_uuid(m.get("document_id"))

        index = m.get("index") or counters.get(document_id, 0)
        language = m.get("language")

        start_char = m.get("start_char")
        end_char = start_char + len(lc_chunk.page_content) if start_char else None

        chunks.append(
            TextChunk(
                id=chunk_id,
                document_id=document_id,
                text=lc_chunk.page_content,
                index=index,
                format=m.get("format") or DocumentFormat.UNKNOWN,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "source": m.get("source"),
                    "language": Language(language) if language else None,
                    "extra": m.get("extra") or {},
                },
            )
        )

        counters[document_id] = index + 1

    return chunks


def from_langchain(lc_chunk: LCDocument) -> TextChunk:
    m = lc_chunk.metadata
    text = lc_chunk.page_content

    chunk_id = parse_uuid(m.get("chunk_id"))
    document_id = parse_uuid(m.get("document_id"))

    language = m.get("language")
    index = m.get("index") or 0

    start_char = m.get("start_index")
    end_char = start_char + len(text) if start_char else None

    return TextChunk(
        id=chunk_id,
        document_id=document_id,
        text=text,
        index=index,
        format=m.get("format") or DocumentFormat.UNKNOWN,
        start_char=start_char,
        end_char=end_char,
        metadata={
            "source": m.get("source"),
            "language": Language(language) if language else None,
            "extra": m.get("extra") or {},
        },
    )


def to_langchain(chunk: TextChunk) -> LCDocument:
    metadata = chunk.metadata or {}

    return LCDocument(
        page_content=chunk.text,
        metadata={
            "document_id": chunk.document_id,
            "index": chunk.index,
            "chunk_id": chunk.id,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "format": chunk.format,

            "source": metadata.get("source"),
            "language": metadata.get("language"),
            "extra": metadata.get("extra"),
        }
    )



"""
LANGCHAIN DOCUMENT

    id: str | None,
    page_content: str,
    metadata: dict[Any, Any]

"""