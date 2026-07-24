from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

ItemT = TypeVar("ItemT")


def filter_by_score(
    items: Sequence[ItemT],
    min_score: float,
    *,
    key: Callable[[ItemT], float],
) -> list[ItemT]:
    return [item for item in items if key(item) >= min_score]


def sort_by_score(
    items: Sequence[ItemT],
    *,
    key: Callable[[ItemT], float],
    reverse: bool = True,
) -> list[ItemT]:
    return sorted(items, key=key, reverse=reverse)
