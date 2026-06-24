from langchain_core.documents import Document as LCDocument
from core.entities.document import Document
from core.entities.chunk import Chunk
from uuid import uuid4, UUID


_METADATA_FIELDS = {
    "chunk_id",
    "document_id",
    "source",
    "title",
    "document_type",
    "language",
    "chunk_index",
}


def to_langchain(chunk: Chunk, parent: Document) -> LCDocument:
    return LCDocument(
        page_content=chunk.content,
        metadata={
            "chunk_id":      str(chunk.id),
            "document_id":   str(parent.id),
            "source":        parent.source,
            "title":         parent.title,
            "document_type": parent.document_type,
            "language":      parent.language,
            "chunk_index":   chunk.index,
            "page_number":   chunk.page_number,
            "section":       chunk.section,
            **chunk.metadata,
        },
    )


def from_langchain(lc_doc: LCDocument) -> Chunk:
    m = lc_doc.metadata or {}

    return Chunk(
        id= _uuid(m.get("chunk_id")),
        document_id= _uuid(m.get("document_id")),
        content= lc_doc.page_content,
        index= m.get("chunk_index", 0),
        metadata={
            key: value
            for key, value in m.items()
            if key not in _METADATA_FIELDS
        },
    )


def _uuid(val) -> UUID:
    try:
        return UUID(str(val))
    except (TypeError, ValueError):
        return uuid4()