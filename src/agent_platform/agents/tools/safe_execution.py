from __future__ import annotations

from collections.abc import AsyncIterator, Coroutine
from typing import Any, TypeVar

from agent_platform.agents.tools.errors import ToolError
from agent_platform.core.errors import PlatformError

ResultT = TypeVar("ResultT")


def _wrap(exc: Exception, error_message: str) -> ToolError:
    """Wrap `exc` in a `ToolError` that keeps the cause's `retryable`.

    Only a `PlatformError` cause says whether a retry can help; any other
    error is a bug or an unknown failure, so it is not retryable.
    """
    retryable = isinstance(exc, PlatformError) and exc.retryable
    return ToolError(f"{error_message}: {exc}", retryable=retryable)


async def safe_call(
    coro: Coroutine[Any, Any, ResultT],
    error_message: str = "Provider execution failed",
) -> ResultT:
    try:
        return await coro
    except ToolError:
        raise
    except Exception as exc:
        raise _wrap(exc, error_message) from exc


async def safe_stream(
    agen: AsyncIterator[ResultT],
    error_message: str = "Provider execution failed",
) -> AsyncIterator[ResultT]:
    try:
        async for item in agen:
            yield item
    except ToolError:
        raise
    except Exception as exc:
        raise _wrap(exc, error_message) from exc
