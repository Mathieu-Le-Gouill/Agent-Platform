from enum import Enum

from agent_platform.core.schemas.config import ProviderConfig


class DistanceMetric(str, Enum):
    COSINE = "cosine"  # angular similarity, scale-invariant; most common default for text embeddings
    DOT = "dot"  # raw dot product; favors vector magnitude, used when embeddings are pre-normalized
    EUCLIDEAN = "euclidean"  # straight-line (L2) distance; smaller is more similar


class VectorStoreConfig(ProviderConfig):
    collection_name: str = "default"  # index/collection identifier to read and write vectors from
    dimension: int | None = None  # embedding vector width; must match the embedding model used to populate the store
    distance: DistanceMetric = DistanceMetric.COSINE  # similarity metric used for nearest-neighbor search; provider enum spellings may differ, see each provider's mapper
    top_k: int = 5  # number of nearest neighbors to return per query
    namespace: str | None = None  # optional logical partition within a collection for multi-tenant isolation
