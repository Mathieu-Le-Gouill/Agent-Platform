from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

In = TypeVar("In", contravariant=True)
Out = TypeVar("Out", covariant=True)


class Component(ABC, Generic[In, Out]):
    @abstractmethod
    async def arun(self, input: In) -> Out: ...

    def run(self, input: In) -> Out:
        import asyncio

        return asyncio.run(self.arun(input))
