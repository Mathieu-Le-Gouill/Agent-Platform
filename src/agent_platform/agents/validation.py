from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from agent_platform.agents.errors import AgentRecoveryExhausted

ResultT = TypeVar("ResultT")

__all__ = ["retry_once_on_invalid"]


async def retry_once_on_invalid(
    *,
    attempt: Callable[[], Awaitable[ResultT]],
    check: Callable[[ResultT], str | None],
    correct: Callable[[str], Awaitable[None]],
) -> ResultT:
    """Run `attempt`, and if `check` reports an error, `correct` it once and retry.

    `check` returns `None` for a valid result, or an error description otherwise.
    `correct` receives that description and should apply whatever corrective
    action (e.g. append feedback to a message list) `attempt` will read on its
    second call. Raises `AgentRecoveryExhausted` (carrying the failing result
    as `last_result`) if the retried attempt is still invalid.
    """
    result = await attempt()
    error = check(result)
    if error is None:
        return result

    await correct(error)

    result = await attempt()
    error = check(result)
    if error is not None:
        raise AgentRecoveryExhausted(error, last_result=result)
    return result
