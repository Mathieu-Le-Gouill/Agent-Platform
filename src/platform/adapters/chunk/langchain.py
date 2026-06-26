from langchain_core.documents import Document as LCDocument
from dataclasses import fields, asdict

from agent_platform.bridges.qdrant.chunk import _uuid
from models.chunk import Chunk, ChunkMetadata


_CHUNK_FIELDS = {f.name for f in fields(Chunk)} - {"metadata"}
_META_FIELDS = {f.name for f in fields(ChunkMetadata)} - {"extra"}
_RESERVED = _CHUNK_FIELDS | {"chunk_id"}


def from_langchain(lc_doc: LCDocument) -> Chunk:
    m = lc_doc.metadata or {}

    return Chunk(
        id= _uuid(m.get("chunk_id")),
        document_id= _uuid(m.get("document_id")),
        text= lc_doc.page_content,
        index= m.get("chunk_index", 0),
        metadata=ChunkMetadata(
            **{k: m.get(k) for k in _META_FIELDS},
            extra={k: v for k, v in m.items() if k not in _META_FIELDS | _RESERVED},
        ),
    )


def to_langchain(chunk: Chunk) -> LCDocument:
    return LCDocument(
        page_content=chunk.text,
        metadata={
            "chunk_id":      str(chunk.id),
            "document_id":   str(chunk.document_id),
            "chunk_index":   chunk.index,

             **{k: v for k, v in asdict(chunk.metadata).items() if k != "extra"},
            **chunk.metadata.extra,
        },
    )
