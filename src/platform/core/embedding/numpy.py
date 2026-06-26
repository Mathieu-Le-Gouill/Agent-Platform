import numpy as np
from models.embedding import Embedding
from uuid import UUID


def to_numpy(embedding: Embedding) -> np.ndarray:
    return np.array(embedding.vector, dtype=np.float32)


def from_numpy(array: np.ndarray, *, model: str, id: UUID) -> Embedding:
    return Embedding.from_list(array.flatten().tolist(), model=model, id=id)