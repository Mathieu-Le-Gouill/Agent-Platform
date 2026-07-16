from enum import Enum

from agent_platform.core.schemas.config import ProviderConfig


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


class VectorStoreConfig(ProviderConfig):
    collection_name: str = "default"
    dimension: int | None = None
    distance: DistanceMetric = DistanceMetric.COSINE
    top_k: int = 5
    namespace: str | None = None
