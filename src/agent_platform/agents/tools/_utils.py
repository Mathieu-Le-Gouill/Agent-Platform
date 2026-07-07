from __future__ import annotations

from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


async def safe_call(
    coro: Coroutine[Any, Any, T],
    error_message: str = "Provider execution failed",
) -> T:
    from agent_platform.agents.tools.base import ToolError

    try:
        return await coro
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"{error_message}: {exc}") from exc
