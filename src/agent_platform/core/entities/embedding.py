from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class Embedding:
    chunk_id: str
    vector: np.ndarray
    model: str