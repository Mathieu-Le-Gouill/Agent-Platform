from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from pydantic import SecretStr

_CompatibleFunc = TypeVar("_CompatibleFunc", bound=Callable[..., Any])
_T = TypeVar("_T")

__all__ = [
    # Exception classes
    "PlatformError",
    "ProviderError",
    "ConfigError",
    "NotFoundError",
    "ValidationError",
    "MissingCredentialError",
    # Utilities
    "require_secret",
    "error_logged",
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


def require_secret(secret: SecretStr | None, message: str) -> SecretStr:
    if secret is None:
        raise MissingCredentialError(message)
    return secret


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
                    return await func(*args, **kwargs)
                except PlatformError:
                    raise  # already a PlatformError: don't re-wrap it in re_raise
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
                    return await func(*args, **kwargs)
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

        if _asyncio_iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorate


@contextmanager
def catch_noraise(
    logger: logging.Logger,
    fallback: _T | None = None,
    context_msg: str = "",
) -> Iterator[None]:
    """Catch any exception, log as warning, yield the fallback value."""
    try:
        yield
    except Exception:
        logger.warning(
            "%s failed — using fallback",
            context_msg or "Operation",
            exc_info=True,
        )


def _asyncio_iscoroutinefunction(func: Callable[..., Any]) -> bool:
    import asyncio

    return asyncio.iscoroutinefunction(func)
