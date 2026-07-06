from dataclasses import dataclass
from enum import Enum


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


@dataclass(slots=True, frozen=True)
class VectorStoreConfig:
    collection_name: str

    dimension: int

    distance: DistanceMetric = DistanceMetric.COSINE

    top_k: int = 5


@dataclass(slots=True, frozen=True)
class ChromaConfig(VectorStoreConfig):
    host: str = "localhost"
    port: int = 8000


@dataclass(slots=True, frozen=True)
class QdrantConfig(VectorStoreConfig):
    url: str = "http://localhost:6333"
