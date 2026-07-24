from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
NewOutputT = TypeVar("NewOutputT")


class Component(ABC, Generic[InputT, OutputT]):
    @abstractmethod
    async def arun(self, input: InputT) -> OutputT: ...

    def run(self, input: InputT) -> OutputT:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(input))
        raise RuntimeError(
            "Component.run() cannot be called from a running event loop; "
            "use 'await component.arun(...)' instead."
        )

    def __rshift__(
        self, other: Component[OutputT, NewOutputT]
    ) -> Component[InputT, NewOutputT]:
        return _Chain(self, other)


class _Chain(Component[InputT, NewOutputT], Generic[InputT, OutputT, NewOutputT]):
    def __init__(
        self, first: Component[InputT, OutputT], second: Component[OutputT, NewOutputT]
    ) -> None:
        self._first = first
        self._second = second

    async def arun(self, input: InputT) -> NewOutputT:
        return await self._second.arun(await self._first.arun(input))
