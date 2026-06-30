from langchain_core.documents import Document as LCDocument
from uuid import UUID
from typing import Sequence

from utils import parse_uuid
from models.chunk import Chunk, ChunkMetadata
from models.enums.language import Language


def from_langchain_many(lc_chunks: Sequence[LCDocument]) -> list[Chunk]:

    counters: dict[UUID, int] = {}

    chunks: list[Chunk] = []

    for lc_chunk in lc_chunks:

        m = lc_chunk.metadata

        chunk_id = parse_uuid(m.get("chunk_id"))
        document_id = parse_uuid(m.get("document_id"))

        index = m.get("index") or counters.get(document_id, 0)
        language = m.get("language")

        start_char = m.get("start_index")
        end_char = start_char + len(lc_chunk.page_content) if start_char else None

        chunk_metadata = ChunkMetadata(
            source=m.get("source"),
            format=m.get("format"),
            language=Language(language) if language else None,
            extra=m.get("extra") or {},
        )

        chunks.append(
            Chunk(
                id=chunk_id,
                document_id=document_id,
                text=lc_chunk.page_content,
                index=index,
                start_char=start_char,
                end_char=end_char,
                metadata=chunk_metadata
            )
        )

        counters[document_id] = index + 1

    return chunks


def from_langchain(lc_chunk: LCDocument) -> Chunk:
    m = lc_chunk.metadata
    text = lc_chunk.page_content

    chunk_id = parse_uuid(m.get("chunk_id"))
    document_id = parse_uuid(m.get("document_id"))

    language = m.get("language")
    index = m.get("index") or 0

    start_char = m.get("start_index")
    end_char = start_char + len(text) if start_char else None

    chunk_metadata = ChunkMetadata(
        source=m.get("source"),
        format=m.get("format"),
        language=Language(language) if language else None,
        extra=m.get("extra") or {},
    )

    return Chunk(
        id=chunk_id,
        document_id=document_id,
        text=text,
        index=index,
        start_char=start_char,
        end_char=end_char,
        metadata=chunk_metadata
    )


def to_langchain(chunk: Chunk) -> LCDocument:
    metadata = chunk.metadata

    return LCDocument(
        page_content=chunk.text,
        metadata={
            "document_id": chunk.document_id,
            "index": chunk.index,
            "chunk_id": chunk.id,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,

            "source": metadata.source,
            "format": metadata.format,
            "language": metadata.language,
            "extra": metadata.extra,

        }
    )



"""
LANGCHAIN DOCUMENT

    id: str | None,
    page_content: str,
    metadata: dict[Any, Any]

"""