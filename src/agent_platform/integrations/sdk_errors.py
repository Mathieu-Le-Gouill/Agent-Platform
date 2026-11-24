"""Classify a vendor SDK failure as a `ProviderError`, shared by every provider.

An SDK raises its own exception types, so the platform decides from what they
carry (an HTTP status, a connection or timeout class) whether a retry can
help. Anything not recognised is permanent, so a bad request or a revoked key
fails at once instead of being retried.
"""

from __future__ import annotations

import functools
import importlib
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

from agent_platform.core.errors import PlatformError, ProviderError

__all__ = ["is_transient_status", "parse_retry_after", "translate_sdk_error"]

_TRANSIENT_STATUS_CODES = frozenset({408, 409, 429})

# Connection and timeout classes of optional SDKs, as `(module, class)`. Each
# is looked up only when its module imports, so none is a dependency.
_OPTIONAL_CONNECTION_TYPES = (
    ("httpx", "TransportError"),
    ("requests", "ConnectionError"),
    ("requests", "Timeout"),
    ("aiohttp", "ClientConnectionError"),
    ("azure.core.exceptions", "ServiceRequestError"),
)

# Attributes that hold an HTTP status on some SDK's error, tried in order.
_STATUS_ATTRIBUTES = (
    "status_code",
    "status",
    "http_status",
    "http_status_code",
    "code",
)


# HTTP status of the same meaning as a gRPC status name; any other name is a
# permanent client-side failure.
_GRPC_STATUS_BY_NAME = {
    "UNAVAILABLE": 503,
    "DEADLINE_EXCEEDED": 504,
    "RESOURCE_EXHAUSTED": 429,
    "ABORTED": 409,
    "INTERNAL": 500,
    "UNKNOWN": 500,
    "DATA_LOSS": 500,
    "UNAUTHENTICATED": 401,
    "PERMISSION_DENIED": 403,
    "NOT_FOUND": 404,
}


def is_transient_status(status_code: int | None) -> bool:
    """Whether an HTTP status marks a temporary failure: a timeout, a conflict, a rate limit or a server error."""
    return status_code is not None and (
        status_code in _TRANSIENT_STATUS_CODES or status_code >= 500
    )


def parse_retry_after(value: str | None) -> float | None:
    """Return the seconds a `Retry-After` header value asks to wait.

    The value is a number of seconds or an HTTP date (RFC 9110 section 10.2.3);
    a date in the past gives `0.0`. Returns `None` when `value` is missing or
    is neither form.
    """
    if value is None:
        return None
    text = value.strip()
    if text.isascii() and text.isdigit():
        return float(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


@functools.cache
def _optional_connection_types() -> tuple[type[BaseException], ...]:
    found: list[type[BaseException]] = []
    for module_name, class_name in _OPTIONAL_CONNECTION_TYPES:
        try:
            cls = getattr(importlib.import_module(module_name), class_name, None)
        except ImportError:
            continue
        if isinstance(cls, type) and issubclass(cls, BaseException):
            found.append(cls)
    return tuple(found)


def _as_status(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _status_from_code_method(exc: BaseException) -> int | None:
    """Read the status of an error whose `code()` is a method.

    Chroma's `code()` returns an HTTP status; gRPC's returns a `StatusCode`,
    which maps to the HTTP status of the same meaning.
    """
    method = getattr(exc, "code", None)
    if not callable(method):
        return None
    try:
        value = method()
    except Exception:
        return None
    status = _as_status(value)
    if status is not None:
        return status
    name = getattr(value, "name", None)
    if not isinstance(name, str):
        return None
    return _GRPC_STATUS_BY_NAME.get(name, 400)


def _status_of(exc: BaseException) -> int | None:
    for attribute in _STATUS_ATTRIBUTES:
        status = _as_status(getattr(exc, attribute, None))
        if status is not None:
            return status
    status = _status_from_code_method(exc)
    if status is not None:
        return status
    response = getattr(exc, "response", None)
    if response is None:
        return None
    if isinstance(response, dict):
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, dict):
            return _as_status(metadata.get("HTTPStatusCode"))
        return None
    return _as_status(getattr(response, "status_code", None))


def _retry_after_of(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    return parse_retry_after(value) if isinstance(value, str) else None


def _chain(exc: BaseException) -> list[BaseException]:
    """Return `exc` and its `__cause__`/`__context__` chain, each once."""
    seen: set[int] = set()
    chain: list[BaseException] = []
    pending: BaseException | None = exc
    while pending is not None and id(pending) not in seen:
        seen.add(id(pending))
        chain.append(pending)
        pending = pending.__cause__ or pending.__context__
    return chain


def _is_connection_error(
    exc: BaseException, transient_types: tuple[type[BaseException], ...]
) -> bool:
    kinds: tuple[type[BaseException], ...] = (
        TimeoutError,
        ConnectionError,
        *transient_types,
        *_optional_connection_types(),
    )
    return any(isinstance(link, kinds) for link in _chain(exc))


def translate_sdk_error(
    exc: Exception,
    *,
    operation: str | None = None,
    transient_types: tuple[type[BaseException], ...] = (),
    retry_statuses: frozenset[int] | None = None,
) -> PlatformError:
    """Return the platform error for a vendor SDK failure.

    A `PlatformError` is returned as is. Otherwise the error is retryable
    when its HTTP status (read from the error or its cause chain, where a gRPC
    status counts as the HTTP status of the same meaning) is transient, when the SDK says so (`should_retry`),
    or when it or its cause is a timeout or a connection error (builtin, in
    `transient_types`, or a class of a known SDK). With `retry_statuses`, only
    a status in that set is retryable: use it where a failure after the
    request left may mean the work was done, such as an image generation.

    The code is `provider_unavailable` for a retryable failure or a connection
    error, `provider_bad_request` for any other failure that carries a status,
    and none for an error with no status, such as a local bug, which the API
    then serves as an internal error.

    Args:
        operation: Prefix of the message, naming what was attempted.
        transient_types: Extra exception types that mean a connection failure.
        retry_statuses: The only statuses worth a retry, or `None` for the
            default rule.
    """
    if isinstance(exc, PlatformError):
        return exc
    status = next((s for s in map(_status_of, _chain(exc)) if s is not None), None)
    connection_error = _is_connection_error(exc, transient_types)
    if retry_statuses is None:
        retryable = (
            is_transient_status(status)
            or getattr(exc, "should_retry", False) is True
            or connection_error
        )
    else:
        retryable = status in retry_statuses
    if retryable or connection_error:
        code: str | None = "provider_unavailable"
    elif status is not None:
        code = "provider_bad_request"
    else:
        code = None
    message = f"{operation}: {exc}" if operation else str(exc)
    return ProviderError(
        message,
        code=code,
        retryable=retryable,
        retry_after=_retry_after_of(exc),
    )
