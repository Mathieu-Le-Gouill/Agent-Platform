from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from agent_platform.core.errors import PlatformError, RecoveryExhausted

_AsyncFunc = TypeVar("_AsyncFunc", bound=Callable[..., Coroutine[Any, Any, Any]])
_T = TypeVar("_T")

__all__ = [
    "MAX_RETRY_AFTER_SECONDS",
    "retry_async",
    "retry_once_on_invalid",
    "retry_sync",
    "with_retry",
]

# Longest wait a service's `retry_after` can impose on one retry. A longer
# request is not honoured in full: a caller should not block for minutes.
MAX_RETRY_AFTER_SECONDS = 30.0


def _next_delay(
    exc: Exception,
    attempt: int,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
) -> float | None:
    """Return the seconds to wait before the next attempt, or `None` to give up.

    An explicitly non-retryable `PlatformError` and the last attempt both give
    up. The delay doubles per attempt up to `max_delay`, with up to 25 % jitter
    so concurrent callers do not retry in lockstep. A `PlatformError` that
    carries `retry_after` raises the delay to that value, capped at
    `MAX_RETRY_AFTER_SECONDS`.
    """
    if isinstance(exc, PlatformError) and not exc.retryable:
        return None
    if attempt == max_attempts:
        return None
    delay = min(max_delay, base_delay * 2 ** (attempt - 1))
    delay *= 1 + random.random() * 0.25
    if isinstance(exc, PlatformError) and exc.retry_after is not None:
        delay = max(delay, min(exc.retry_after, MAX_RETRY_AFTER_SECONDS))
    return delay


def _log_retry(
    logger: logging.Logger, name: str, attempt: int, max_attempts: int, delay: float
) -> None:
    logger.warning(
        "%s failed (attempt %d/%d), retrying in %.2fs",
        name,
        attempt,
        max_attempts,
        delay,
    )


async def retry_async(
    attempt_once: Callable[[], Coroutine[Any, Any, _T]],
    *,
    max_attempts: int,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> _T:
    """Await `attempt_once()` up to `max_attempts` times, with exponential backoff.

    Retries an exception in `retry_on` unless it is a `PlatformError` with
    `retryable=False`, which is raised at once. The last failure is raised as is.

    Args:
        attempt_once: Builds a fresh coroutine for each attempt.
        max_attempts: Total number of calls, at least 1.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    logger = logging.getLogger(getattr(attempt_once, "__module__", __name__))
    name = getattr(attempt_once, "__qualname__", "call")
    attempt = 1
    while True:
        try:
            return await attempt_once()
        except retry_on as exc:
            delay = _next_delay(exc, attempt, max_attempts, base_delay, max_delay)
            if delay is None:
                raise
            _log_retry(logger, name, attempt, max_attempts, delay)
            await asyncio.sleep(delay)
            attempt += 1


def retry_sync(
    attempt_once: Callable[[], _T],
    *,
    max_attempts: int,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> _T:
    """Blocking counterpart of `retry_async`, sleeping with `time.sleep`."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    logger = logging.getLogger(getattr(attempt_once, "__module__", __name__))
    name = getattr(attempt_once, "__qualname__", "call")
    attempt = 1
    while True:
        try:
            return attempt_once()
        except retry_on as exc:
            delay = _next_delay(exc, attempt, max_attempts, base_delay, max_delay)
            if delay is None:
                raise
            _log_retry(logger, name, attempt, max_attempts, delay)
            time.sleep(delay)
            attempt += 1


def with_retry(
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    translate: Callable[[Exception], Exception] | None = None,
) -> Callable[[_AsyncFunc], _AsyncFunc]:
    """Retry the decorated coroutine function as `retry_async` does.

    With `translate`, each failed attempt's exception is translated first, so
    the retry decision reads the translated error (for example a `PlatformError`
    with `retryable=False` stops at once). A new error is raised `from` the
    original; one `translate` returns unchanged is re-raised as it is.
    `asyncio.CancelledError` is never translated.
    """

    def decorate(func: _AsyncFunc) -> _AsyncFunc:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            call = functools.partial(func, *args, **kwargs)
            if translate is not None:

                async def call_translated() -> Any:
                    try:
                        return await call()
                    except Exception as exc:
                        translated = translate(exc)
                        if translated is exc:
                            raise
                        raise translated from exc

                run: Callable[[], Coroutine[Any, Any, Any]] = call_translated
            else:
                run = call
            # Carries the decorated function's module and name, which the
            # retry log lines use.
            attempt_once = functools.update_wrapper(run, func)
            return await retry_async(
                attempt_once,
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                retry_on=retry_on,
            )

        return wrapper  # type: ignore[return-value]

    return decorate


async def retry_once_on_invalid(
    *,
    attempt: Callable[[], Awaitable[_T]],
    check: Callable[[_T], str | None],
    correct: Callable[[str], Awaitable[None]],
    on_retry: Callable[[], None] | None = None,
) -> _T:
    """Run `attempt`, and if `check` reports an error, `correct` it once and retry.

    `check` returns `None` for a valid result, or an error description otherwise.
    `correct` receives that description and should apply whatever corrective
    action (e.g. append feedback to a message list) `attempt` will read on its
    second call. Raises `RecoveryExhausted` (carrying the failing result as
    `last_result`) if the retried attempt is still invalid. `on_retry` is
    notified when a retry happens, so a caller can observe recovery without
    inspecting the results itself.
    """
    result = await attempt()
    error = check(result)
    if error is None:
        return result

    if on_retry is not None:
        on_retry()
    await correct(error)

    result = await attempt()
    error = check(result)
    if error is not None:
        raise RecoveryExhausted(error, last_result=result)
    return result
