from __future__ import annotations

from pydantic import BaseModel


class Dimensions(BaseModel, frozen=True):
    width: int
    height: int
    depth: int | None = None


_MODE_BIT_DEPTH: dict[str, int] = {
    "1": 1,
    "L": 8,
    "LA": 8,
    "P": 8,
    "I": 32,
    "F": 32,
    "RGB": 8,
    "RGBA": 8,
    "CMYK": 8,
    "YCbCr": 8,
    "LAB": 8,
    "HSV": 8,
    "I;16": 16,
    "I;16L": 16,
    "I;16B": 16,
}


def bit_depth_for_mode(mode: str) -> int | None:
    return _MODE_BIT_DEPTH.get(mode)
