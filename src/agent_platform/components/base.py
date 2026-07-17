from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

In = TypeVar("In", contravariant=True)
Out = TypeVar("Out", covariant=True)
NewOut = TypeVar("NewOut")


class Component(ABC, Generic[In, Out]):
    @abstractmethod
    async def arun(self, input: In) -> Out: ...

    def run(self, input: In) -> Out:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(input))
        raise RuntimeError(
            "Component.run() cannot be called from a running event loop; "
            "use 'await component.arun(...)' instead."
        )

    def __rshift__(self, other: Component[Out, NewOut]) -> Component[In, NewOut]:
        return _Chain(self, other)


class _Chain(Component[In, NewOut], Generic[In, Out, NewOut]):
    def __init__(self, first: Component[In, Out], second: Component[Out, NewOut]) -> None:
        self._first = first
        self._second = second

    async def arun(self, input: In) -> NewOut:
        return await self._second.arun(await self._first.arun(input))
