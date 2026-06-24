from dataclasses import dataclass, field
import numpy as np
from uuid import UUID
from typing import Optional

@dataclass(slots=True)
class Embedding:
    chunk_id: Optional[UUID] = None
    vector: np.ndarray = field(default_factory=lambda: np.array([]))
    model: str = ""