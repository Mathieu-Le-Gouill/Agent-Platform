from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from agent_platform.core.errors import PlatformError

_AsyncFunc = TypeVar("_AsyncFunc", bound=Callable[..., Coroutine[Any, Any, Any]])

__all__ = ["with_retry"]


def with_retry(
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[_AsyncFunc], _AsyncFunc]:
    def decorate(func: _AsyncFunc) -> _AsyncFunc:
        logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as exc:
                    if isinstance(exc, PlatformError) and not exc.retryable:
                        raise
                    if attempt == max_attempts:
                        raise
                    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
                    delay *= 1 + random.random() * 0.25
                    logger.warning(
                        "%s failed (attempt %d/%d), retrying in %.2fs",
                        func.__qualname__,
                        attempt,
                        max_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)

        return wrapper  # type: ignore[return-value]

    return decorate
