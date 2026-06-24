import numpy as np
from core.entities.embedding import Embedding


def to_ndarray(embedding: Embedding) -> np.ndarray:
    return np.array(embedding.vector, dtype=np.float32)


def from_ndarray(array: np.ndarray, *, model: str) -> Embedding:
    return Embedding.from_list(array.flatten().tolist(), model=model)