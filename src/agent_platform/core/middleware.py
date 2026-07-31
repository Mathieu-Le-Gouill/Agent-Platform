from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Generic, Protocol, TypeVar

CtxT = TypeVar("CtxT")

__all__ = ["Middleware", "MiddlewarePipeline"]


class Middleware(Protocol[CtxT]):
    async def before(self, ctx: CtxT) -> CtxT | None: ...

    async def after(self, ctx: CtxT, result: Any) -> Any: ...


class MiddlewarePipeline(Generic[CtxT]):
    def __init__(self, middlewares: list[Middleware[CtxT]]) -> None:
        self._middlewares = middlewares

    async def run(self, ctx: CtxT, operation: Callable[[CtxT], Awaitable[Any]]) -> Any:
        for middleware in self._middlewares:
            short_circuit = await middleware.before(ctx)
            if short_circuit is not None:
                return short_circuit

        result = await operation(ctx)

        for middleware in reversed(self._middlewares):
            result = await middleware.after(ctx, result)

        return result
