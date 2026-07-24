from __future__ import annotations

import math
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class Embedding(BaseModel, frozen=True):
    vector: tuple[float, ...]
    id: UUID = Field(default_factory=uuid4)
    model: str = ""

    @model_validator(mode="after")
    def _check_not_empty(self) -> Embedding:
        if len(self.vector) == 0:
            raise ValueError("Embedding vector cannot be empty")
        return self

    @classmethod
    def from_list(
        cls, vector: list[float], model: str = "", id: UUID | None = None
    ) -> Embedding:
        return cls(vector=tuple(vector), model=model, id=id or uuid4())

    def to_list(self) -> list[float]:
        return list(self.vector)

    @property
    def dimensions(self) -> int:
        return len(self.vector)

    @property
    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.vector))

    def is_normalized(self, tol: float = 1e-4) -> bool:
        return abs(self.norm - 1.0) < tol
