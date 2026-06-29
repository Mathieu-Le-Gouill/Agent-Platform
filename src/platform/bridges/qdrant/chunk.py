from dataclasses import fields, asdict

from qdrant_client.models import PointStruct

from utils import parse_uuid
from models.chunk import Chunk, ChunkMetadata
from models.score import Score, ScoreKind


_META_FIELDS = {f.name for f in fields(ChunkMetadata)} - {"extra"}
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
        id=          parse_uuid(p.get("chunk_id") or point.id),
        document_id= parse_uuid(p.get("document_id")),
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


def chunks_from_points(points: list) -> list[Chunk]:
    return [from_point(point) for point in points]


# --- Helpers ---


def _chunk_payload(chunk: Chunk) -> dict:
    return {
        "chunk_id":    str(chunk.id),
        "document_id": str(chunk.document_id),
        "chunk_index": chunk.index,
        "text":        chunk.text,

        **{k: v for k, v in asdict(chunk.metadata).items() if k != "extra"},
        **chunk.metadata.extra,
    }
    

def _distance_to_similarity(point_score: float) -> float:
    return max(-1.0, min(1.0, point_score))