from __future__ import annotations

from typing import Generic, Protocol, TypeVar

StateT = TypeVar("StateT")

__all__ = ["Checkpointer", "InMemoryCheckpointer"]


class Checkpointer(Protocol[StateT]):
    async def save(self, key: str, state: StateT) -> None: ...

    async def load(self, key: str) -> StateT | None: ...


class InMemoryCheckpointer(Generic[StateT]):
    def __init__(self) -> None:
        self._store: dict[str, StateT] = {}

    async def save(self, key: str, state: StateT) -> None:
        self._store[key] = state

    async def load(self, key: str) -> StateT | None:
        return self._store.get(key)
