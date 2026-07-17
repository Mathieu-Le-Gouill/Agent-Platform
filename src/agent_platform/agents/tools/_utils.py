from __future__ import annotations

from typing import Any, AsyncIterator, Coroutine, TypeVar

from agent_platform.agents.tools.errors import ToolError

T = TypeVar("T")


async def safe_call(
    coro: Coroutine[Any, Any, T],
    error_message: str = "Provider execution failed",
) -> T:
    try:
        return await coro
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"{error_message}: {exc}", retryable=True) from exc


async def safe_stream(
    agen: AsyncIterator[T],
    error_message: str = "Provider execution failed",
) -> AsyncIterator[T]:
    try:
        async for item in agen:
            yield item
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"{error_message}: {exc}", retryable=True) from exc
