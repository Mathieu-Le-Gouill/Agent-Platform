from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Generic, Protocol, TypeVar

CtxT = TypeVar("CtxT", contravariant=True)
ResultT = TypeVar("ResultT")

__all__ = ["Middleware", "MiddlewarePipeline"]


class Middleware(Protocol[CtxT, ResultT]):
    async def before(self, ctx: CtxT) -> ResultT | None: ...

    async def after(self, ctx: CtxT, result: ResultT) -> ResultT: ...


class MiddlewarePipeline(Generic[CtxT, ResultT]):
    def __init__(self, middlewares: list[Middleware[CtxT, ResultT]]) -> None:
        self._middlewares = middlewares

    async def run(
        self, ctx: CtxT, operation: Callable[[CtxT], Awaitable[ResultT]]
    ) -> ResultT:
        for middleware in self._middlewares:
            short_circuit = await middleware.before(ctx)
            if short_circuit is not None:
                return short_circuit

        result = await operation(ctx)

        for middleware in reversed(self._middlewares):
            result = await middleware.after(ctx, result)

        return result
