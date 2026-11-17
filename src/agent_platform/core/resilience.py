from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, TypeVar

from agent_platform.core.errors import ConfigError, PlatformError, ProviderError

_T = TypeVar("_T")

__all__ = ["CircuitBreaker", "CircuitState", "RateLimiter"]


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._logger = logging.getLogger(__name__)

    @property
    def state(self) -> CircuitState:
        # lazily flips OPEN -> HALF_OPEN on read once reset_timeout has elapsed
        if (
            self._state is CircuitState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self._reset_timeout
        ):
            self._state = CircuitState.HALF_OPEN
        return self._state

    def call(self, func: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        if self.state is CircuitState.OPEN:
            raise ProviderError("Circuit breaker is open", retryable=True)

        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            self.record_error(exc)
            raise
        else:
            self.record_success()
            return result

    async def acall(
        self, func: Callable[..., Awaitable[_T]], *args: Any, **kwargs: Any
    ) -> _T:
        if self.state is CircuitState.OPEN:
            raise ProviderError("Circuit breaker is open", retryable=True)

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            self.record_error(exc)
            raise
        else:
            self.record_success()
            return result

    def record_error(self, exc: Exception) -> None:
        """Count `exc` as a failure unless it is a non-retryable `PlatformError`.

        A non-retryable error (a bad request, a revoked key) says nothing about
        whether the service is up, so it must not open the circuit. Public for
        call sites that can't route through `call()`/`acall()`, such as one
        step of a streaming response.
        """
        if isinstance(exc, PlatformError) and not exc.retryable:
            return
        self.record_failure()

    def record_success(self) -> None:
        """Reset the breaker to closed. Public so callers that can't route a
        call through `call()`/`acall()` (e.g. one step of a streaming
        response) can still report the outcome, e.g. `FallbackLLMProvider`."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            if self._state is not CircuitState.OPEN:
                self._logger.warning(
                    "Circuit breaker opening after %d consecutive failures",
                    self._failure_count,
                )
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()


class RateLimiter:
    """Token-bucket limiter: `rate` tokens/second replenish up to `burst` capacity.

    Raises `ConfigError` when `rate <= 0` or the capacity (`burst`, else `rate`)
    is below 1: a bucket that can never hold one token would make `acquire()`
    wait forever.
    """

    def __init__(self, *, rate: float, burst: float | None = None) -> None:
        capacity = burst if burst is not None else rate
        if rate <= 0:
            raise ConfigError(f"rate must be greater than 0, got {rate}")
        if capacity < 1:
            raise ConfigError(
                f"burst (default: rate) must be at least 1 so one token fits, "
                f"got {capacity}"
            )
        self._rate = rate
        self._capacity = capacity
        self._tokens = self._capacity
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            while True:
                self._replenish()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                await asyncio.sleep(deficit / self._rate)

    def _replenish(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated_at
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._updated_at = now
