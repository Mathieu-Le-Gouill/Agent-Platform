from __future__ import annotations

from math import gcd

__all__ = ["size_to_aspect"]


def size_to_aspect(size: str | None) -> str:
    if size is None:
        return "1:1"
    parts = size.lower().split("x")
    if len(parts) != 2:
        return "1:1"
    try:
        w, h = int(parts[0]), int(parts[1])
        g = gcd(w, h)
        return f"{w // g}:{h // g}"
    except (ValueError, ZeroDivisionError):
        return "1:1"
