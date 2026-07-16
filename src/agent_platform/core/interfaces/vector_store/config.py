from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


class VectorStoreConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    collection_name: str = "default"
    dimension: int | None = None
    distance: DistanceMetric = DistanceMetric.COSINE
    top_k: int = 5
    namespace: str | None = None
