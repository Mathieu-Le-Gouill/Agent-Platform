from uuid import UUID, uuid4
import dataclasses

from qdrant_client.models import PointStruct

from models.chunk import Chunk, ChunkMetadata
from models.score import Score, ScoreKind


_META_FIELDS = {f.name for f in dataclasses.fields(ChunkMetadata)} - {"extra"}
_RESERVED    = _META_FIELDS | {"chunk_id", "document_id", "chunk_index"}


def to_point(chunk: Chunk) -> PointStruct:
    if chunk.embedding is None:
        raise ValueError("Embedding vector is None")

    return PointStruct(
        id=str(chunk.id),
        vector=chunk.embedding,
        payload=_chunk_payload(chunk),
    )


def from_point(point) -> Chunk:
    p = point.payload

    chunk = Chunk(
        id=          _uuid(p.get("chunk_id") or point.id),
        document_id= _uuid(p.get("document_id")),
        text=        p.get("text", ""),
        index=       p.get("chunk_index", 0),

        metadata=ChunkMetadata(
            **{k: p.get(k) for k in _META_FIELDS},
            extra={k: v for k, v in p.items() if k not in _META_FIELDS | _RESERVED | {"text"}},
        ),
    )

    if point.score is not None:
            chunk.scores[ScoreKind.SIMILARITY] = Score.similarity(_distance_to_similarity(point.score))

    return chunk


# --- Helpers ---


def _chunk_payload(chunk: Chunk) -> dict:
    return {
        "chunk_id":    str(chunk.id),
        "document_id": str(chunk.document_id),
        "chunk_index": chunk.index,
        "text":        chunk.text,

        **{k: v for k, v in dataclasses.asdict(chunk.metadata).items() if k != "extra"},
        **chunk.metadata.extra,
    }


def _uuid(val) -> UUID:
    try:
        return UUID(str(val))
    except (TypeError, ValueError):
        return uuid4()
    

def _distance_to_similarity(point_score: float) -> float:
    return max(-1.0, min(1.0, point_score))