from langchain_core.documents import Document as LCDocument
from agent_platform.models.document import Document
from agent_platform.models.chunk import Chunk, ChunkMetadata

import dataclasses
from uuid import uuid4, UUID

_METADATA_FIELDS = {f.name for f in dataclasses.fields(ChunkMetadata)} - {"extra"}


def to_langchain(chunk: Chunk, parent: Document) -> LCDocument:
    return LCDocument(
        page_content=chunk.content,
        metadata={
            "chunk_id":      str(chunk.id),
            "document_id":   str(parent.id),
            "chunk_index":   chunk.index,

            # The chunk metadata
            **{k: v for k, v in dataclasses.asdict(chunk.metadata).items() if k != "extra"},
            **chunk.metadata.extra,
        },
    )


def from_langchain(lc_doc: LCDocument) -> Chunk:
    m = lc_doc.metadata or {}
    known = {f.name for f in dataclasses.fields(ChunkMetadata)}

    return Chunk(
        id= _uuid(m.get("chunk_id")),
        document_id= _uuid(m.get("document_id")),
        content= lc_doc.page_content,
        index= m.get("chunk_index", 0),
        metadata=ChunkMetadata(
            **{k: m.get(k) for k in known if k != "extra"},
            extra={k: v for k, v in m.items() if k not in known | {"chunk_id", "document_id", "chunk_index"}},
        ),
    )


def _uuid(val) -> UUID:
    try:
        return UUID(str(val))
    except (TypeError, ValueError):
        return uuid4()