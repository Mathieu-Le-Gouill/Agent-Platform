from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from agent_platform.core.observability.facts import Observation
from agent_platform.core.observability.runtime import observe

__all__ = ["traced"]

P = ParamSpec("P")
R = TypeVar("R")


def traced(
    observation: type[Observation], /, **facts: Any
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Run a function inside `observe(observation(**facts))`.

    For call sites whose whole body is the operation (a pipeline entry point,
    a component's `arun`), where wrapping is all the instrumentation needed.
    Works on sync and async functions alike. A fresh observation is built per
    call, since an observation is mutable; anything only known at runtime
    belongs on `current(...)` inside the body instead.
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                with observe(observation(**facts)):
                    return await fn(*args, **kwargs)

            return cast(Callable[P, R], async_wrapper)

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with observe(observation(**facts)):
                return fn(*args, **kwargs)

        return wrapper

    return decorate
