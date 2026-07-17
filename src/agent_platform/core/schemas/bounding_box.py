from __future__ import annotations

from pydantic import BaseModel


class BoundingBox(BaseModel, frozen=True):
    x: float
    y: float
    width: float
    height: float
    normalized: bool = False
