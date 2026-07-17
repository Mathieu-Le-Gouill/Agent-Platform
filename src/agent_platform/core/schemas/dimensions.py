from __future__ import annotations

from pydantic import BaseModel


class Dimensions(BaseModel, frozen=True):
    width: int
    height: int
    depth: int | None = None
