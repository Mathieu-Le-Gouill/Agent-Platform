from __future__ import annotations

from dataclasses import dataclass

from agent_platform.core.errors import PlatformError

__all__ = ["ErrorResponse", "INTERNAL_ERROR", "describe_error"]

INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ErrorResponse:
    """What the API tells a client about a failure: an HTTP status and a stable body."""

    status: int
    error: str
    message: str


# The messages are fixed text on purpose: an exception's own text can echo the
# prompt or a vendor's reply, and must never reach a client.
_BY_CODE: dict[str, ErrorResponse] = {
    response.error: response
    for response in (
        ErrorResponse(
            429,
            "llm_rate_limited",
            "The model provider is rate limiting requests; retry later",
        ),
        ErrorResponse(504, "llm_timeout", "The model provider did not answer in time"),
        ErrorResponse(
            422, "llm_context_window", "The conversation is too long for the model"
        ),
        ErrorResponse(
            503,
            "provider_unavailable",
            "The model provider is temporarily unavailable; retry later",
        ),
        ErrorResponse(
            502,
            "provider_bad_request",
            "The model provider rejected the request",
        ),
        ErrorResponse(
            502,
            "agent_recovery_exhausted",
            "The agent could not produce a valid response",
        ),
        ErrorResponse(
            500,
            "agent_max_iterations",
            "The agent used all its iterations without an answer",
        ),
    )
}
_INTERNAL = ErrorResponse(500, INTERNAL_ERROR, "Internal server error")


def describe_error(exc: BaseException) -> ErrorResponse:
    """Return the API response for a failure, by the error's `code`.

    A `PlatformError` with an unlisted code and any other exception give the
    generic 500 `internal_error`.
    """
    if isinstance(exc, PlatformError) and exc.code is not None:
        return _BY_CODE.get(exc.code, _INTERNAL)
    return _INTERNAL
