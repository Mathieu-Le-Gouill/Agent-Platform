from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(slots=True)
class Embedding:
    vector: tuple[float, ...]
    model: str = ""
    dimensions: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "dimensions", len(self.vector))
        if self.dimensions == 0:
            raise ValueError("Embedding vector cannot be empty")

    @classmethod
    def from_list(cls, vector: list[float], model: str) -> Embedding:
        return cls(vector=tuple(vector), model=model)

    def to_list(self) -> list[float]:
        return list(self.vector)

    @property
    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.vector))

    def is_normalized(self, tol: float = 1e-4) -> bool:
        return abs(self.norm - 1.0) < tol