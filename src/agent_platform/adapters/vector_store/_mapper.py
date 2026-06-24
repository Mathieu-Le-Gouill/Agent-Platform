
from uuid import UUID

from qdrant_client.models import PointStruct

from core.entities.chunk import Chunk
from core.entities.document import Document


# --- Qdrant ---

def chunk_to_point(chunk: Chunk, document: Document) -> PointStruct:
    if chunk.embedding is None:
        raise ValueError("Embedding vector is None")

    return PointStruct(
        id=str(chunk.id),
        vector=chunk.embedding,
        payload={
            "content":     chunk.content,
            "document_id": str(document.id),
            "index":       chunk.index,
            "metadata":    document.metadata,
        },
    )


def point_to_chunk(point) -> Chunk:
    p = point.payload
    return Chunk(
        id=UUID(point.id),
        content=p["content"],
        index=p["index"],
        document_id=UUID(p["document_id"]),
        metadata=p.get("metadata", {}),
    )


# --- Chroma ---

def chunk_to_chroma(chunk: Chunk, document: Document) -> dict:
    return {
        "id":        str(chunk.id),
        "embedding": chunk.embedding,
        "content":   chunk.content,
        "metadata": {
            "document_id": str(document.id),
            "index":       chunk.index,
            **{k: v for k, v in document.metadata.items() if isinstance(v, (str, int, float, bool))},
        },
    }


def chroma_to_chunk(content: str, metadata: dict) -> Chunk:
    return Chunk(
        id=UUID(metadata["id"]) if "id" in metadata else UUID(int=0),
        content=content,
        index=metadata.get("index", 0),
        document_id=UUID(metadata["document_id"]),
        metadata={k: v for k, v in metadata.items() if k not in {"document_id", "index"}},
    )