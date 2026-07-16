from __future__ import annotations

import asyncio
import functools
import logging
import random
from contextlib import contextmanager
from typing import Any, Callable, Coroutine, TypeVar

_CompatibleFunc = TypeVar("_CompatibleFunc", bound=Callable[..., Any])
_AsyncFunc = TypeVar("_AsyncFunc", bound=Callable[..., Coroutine[Any, Any, Any]])

__all__ = [
    # Exception classes
    "PlatformError",
    "ProviderError",
    "ConfigError",
    "NotFoundError",
    "ValidationError",
    "MissingCredentialError",
    # Utilities
    "error_logged",
    "with_retry",
    "catch_noraise",
]


class PlatformError(Exception):
    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        retryable: bool = False,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.context = context or {}
        super().__init__(message)


class ProviderError(PlatformError):
    pass


class ConfigError(PlatformError):
    pass


class NotFoundError(PlatformError):
    pass


class ValidationError(PlatformError):
    pass


class MissingCredentialError(PlatformError):
    pass


# --- Utilities ---


def error_logged(
    logger_name: str | None = None,
    *,
    re_raise: type[PlatformError] | None = None,
    message: str = "Unhandled error",
) -> Callable[[_CompatibleFunc], _CompatibleFunc]:
    def decorate(func: _CompatibleFunc) -> _CompatibleFunc:
        logger = logging.getLogger(logger_name or func.__module__)

        if re_raise is not None:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)  # type: ignore
                except PlatformError:
                    raise
                except Exception as exc:
                    logger.exception("%s in %s", message, func.__qualname__)
                    raise re_raise(f"{message}: {exc}") from exc

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except PlatformError:
                    raise
                except Exception as exc:
                    logger.exception("%s in %s", message, func.__qualname__)
                    raise re_raise(f"{message}: {exc}") from exc

        else:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)  # type: ignore
                except PlatformError:
                    raise
                except Exception:
                    logger.exception("%s in %s", message, func.__qualname__)
                    raise

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except PlatformError:
                    raise
                except Exception:
                    logger.exception("%s in %s", message, func.__qualname__)
                    raise

        if asyncio_iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorate


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
                except retry_on:
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

        return wrapper  # type: ignore

    return decorate


@contextmanager
def catch_noraise(
    logger: logging.Logger,
    fallback: Any = None,
    context_msg: str = "",
):
    """Catch any exception, log as warning, yield the fallback value."""
    try:
        yield
    except Exception:
        logger.warning(
            "%s failed — using fallback",
            context_msg or "Operation",
            exc_info=True,
        )


def asyncio_iscoroutinefunction(func: Callable[..., Any]) -> bool:
    import asyncio

    return asyncio.iscoroutinefunction(func)
