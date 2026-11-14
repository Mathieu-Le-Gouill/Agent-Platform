from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from agent_platform.core.observability import ComponentRun, traced

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
NewOutputT = TypeVar("NewOutputT")

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _default_operation_name(cls: type) -> str:
    # Leading underscores are a Python visibility marker, not part of the name
    # an operator reads in a trace.
    return _CAMEL_BOUNDARY.sub("_", cls.__name__.lstrip("_")).lower()


class Component(ABC, Generic[InputT, OutputT]):
    #: Span/operation name for one `arun`. Defaults to the class name in
    #: snake_case; override where a standard name exists (`Embedder` reports
    #: `embeddings`, a GenAI semantic-convention operation).
    operation_name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only wrap a subclass's own `arun`: an inherited one is already
        # traced by the class that defined it, so this never nests twice.
        own_arun = cls.__dict__.get("arun")
        if own_arun is None:
            return
        operation = cls.operation_name or _default_operation_name(cls)
        cls.arun = traced(ComponentRun, name=operation)(own_arun)  # type: ignore[method-assign]

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
