from __future__ import annotations

from typing import Iterator, Sequence, TypeVar

__all__ = ["chunked"]

T = TypeVar("T")


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
