from __future__ import annotations

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def filter_by_score(
    items: Sequence[T],
    min_score: float,
    *,
    key: Callable[[T], float],
) -> list[T]:
    return [item for item in items if key(item) >= min_score]


def sort_by_score(
    items: Sequence[T],
    *,
    key: Callable[[T], float],
    reverse: bool = True,
) -> list[T]:
    return sorted(items, key=key, reverse=reverse)
