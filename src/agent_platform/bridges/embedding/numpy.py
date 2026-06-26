import numpy as np
from models.embedding import Embedding


def embedding_to_ndarray(embedding: Embedding) -> np.ndarray:
    return np.array(embedding.vector, dtype=np.float32)


def embedding_from_ndarray(array: np.ndarray, *, model: str) -> Embedding:
    return Embedding.from_list(array.flatten().tolist(), model=model)