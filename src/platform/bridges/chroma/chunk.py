import dataclasses

from models.chunk import Chunk, ChunkMetadata
from models.score import Score, ScoreKind
from utils import parse_uuid

_META_FIELDS = {f.name for f in dataclasses.fields(ChunkMetadata)} - {"extra"}
_RESERVED    = _META_FIELDS | {"chunk_id", "document_id", "chunk_index"}


def to_chroma(chunk: Chunk) -> dict:
    return {
        "id":        str(chunk.id),
        "embedding": chunk.embedding,
        "text":      chunk.text,
        "metadata": _chunk_metadata(chunk),
    }


def to_chroma_many(chunks: list[Chunk]) -> dict:
    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    for chunk in chunks:
        ids.append(str(chunk.id))
        embeddings.append(chunk.embedding)
        documents.append(chunk.text)
        metadatas.append(_chunk_metadata(chunk))

    return {
        "ids": ids,
        "embeddings": embeddings,
        "documents": documents,
        "metadatas": metadatas,
    }


def from_chroma(text: str, metadata: dict, distance: float | None = None) -> Chunk:
    chunk = Chunk(
        id=          parse_uuid(metadata.get("chunk_id")),
        document_id= parse_uuid(metadata.get("document_id")),
        text=        text,
        index=       metadata.get("chunk_index", 0),
        metadata=    ChunkMetadata(
            **{k: metadata.get(k) for k in _META_FIELDS},
            extra={k: v for k, v in metadata.items() if k not in _META_FIELDS | _RESERVED},
        ),
    )

    if distance is not None:
        # Chroma returns L2 distance by default - lower is better, convert to similarity
        chunk.scores[ScoreKind.SIMILARITY] = Score.similarity(_distance_to_similarity(distance))

    return chunk


def from_chroma_many(results: dict) -> list[Chunk]:
    chunks    = []
    documents = results["documents"]
    metadatas = results["metadatas"]
    distances = results.get("distances") or [None] * len(documents)

    for text, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(from_chroma(text, metadata, distance))

    return chunks


# --- Helpers ---

def _chunk_metadata(chunk: Chunk) -> dict:
    return {
        "chunk_id":    str(chunk.id),
        "document_id": str(chunk.document_id),
        "chunk_index": chunk.index,
        **{
            k: v
            for k, v in dataclasses.asdict(chunk.metadata).items()
            if k != "extra" and isinstance(v, (str, int, float, bool))
        },
        **{
            k: v
            for k, v in chunk.metadata.extra.items()
            if isinstance(v, (str, int, float, bool))
        },
    }


def _distance_to_similarity(distance: float) -> float:
    # Chroma with hnsw:space=cosine returns cosine distance in [0, 2]
    # cosine_similarity = 1 - cosine_distance, clamped to [-1, 1]
    return max(-1.0, min(1.0, 1.0 - distance))