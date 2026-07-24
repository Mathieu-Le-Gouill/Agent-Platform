from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

__all__ = ["chunked"]

ItemT = TypeVar("ItemT")


def chunked(items: Sequence[ItemT], size: int) -> Iterator[Sequence[ItemT]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
