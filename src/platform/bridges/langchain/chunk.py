from langchain_core.documents import Document as LCDocument
from uuid import UUID

from utils import parse_uuid
from models.chunk import Chunk, ChunkMetadata
from models.enums.language import Language


def from_langchain(lc_chunks: list[LCDocument]) -> list[Chunk]:

    counters: dict[UUID, int] = {}

    chunks: list[Chunk] = []

    for lc_chunk in lc_chunks:

        m = lc_chunks[0].metadata if lc_chunks else {}

        document_id = parse_uuid(m.get("document_id"))

        index = counters.get(document_id, 0)
        language = m.get("language")

        start_char = lc_chunk.metadata.get("start_index")
        end_char = start_char + len(lc_chunk.page_content) if start_char else None

        chunk_metadata = ChunkMetadata(
            source=m.get("source"),
            format=m.get("format"),
            language=Language(language) if language else None,
            extra=m.get("extra") or {},
        )

        chunks.append(
            Chunk(
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


"""
LANGCHAIN DOCUMENT

    id: str | None,
    page_content: str,
    metadata: dict[Any, Any]

"""