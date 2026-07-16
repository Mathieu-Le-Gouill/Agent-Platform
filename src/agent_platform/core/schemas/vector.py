from __future__ import annotations

from pydantic import BaseModel, model_validator


class SparseVector(BaseModel, frozen=True):
    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def _validate_lengths(self) -> "SparseVector":
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"SparseVector indices/values length mismatch: "
                f"{len(self.indices)} != {len(self.values)}"
            )
        return self
