from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from langfuse import Langfuse

from agent_platform.core.interfaces.tracing.base import BaseTracer
from agent_platform.core.interfaces.tracing.noop import NoOpSpan, NoOpTracer
from agent_platform.core.interfaces.tracing.span import AttributeValue, Span
from agent_platform.core.observability import ERROR_TYPE_KEY
from agent_platform.integrations.tracing.langfuse.mapping import (
    ObservationType,
    observation_type,
    update_fields,
)

if TYPE_CHECKING:
    from langfuse._client.span import LangfuseObservationWrapper

__all__ = ["LangfuseSpan", "LangfuseTracer"]

logger = logging.getLogger(__name__)

_REQUIRED_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

_NOOP_SPAN = NoOpSpan()


class LangfuseSpan:
    """Adapts a Langfuse observation to the platform's `Span` protocol."""

    __slots__ = ("_as_type", "_observation", "_usage")

    def __init__(
        self,
        observation: LangfuseObservationWrapper,
        as_type: ObservationType,
        usage: Mapping[str, int] | None = None,
    ) -> None:
        self._observation = observation
        self._as_type = as_type
        # Usage arrives over several `set_attribute` calls and the SDK
        # replaces the whole `usage_details` value on each update, so the
        # running total is kept here and always sent in full.
        self._usage: dict[str, int] = dict(usage or {})

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self._update({key: value})

    def record_error(self, exc: Exception, *, record_exception: bool = True) -> None:
        # The SDK has no public stack-trace API, so the exception type goes in
        # the status message when the caller wants the exception recorded.
        message = f"{type(exc).__qualname__}: {exc}" if record_exception else str(exc)
        self._observation.update(
            level="ERROR",
            status_message=message,
            metadata={ERROR_TYPE_KEY: type(exc).__qualname__},
        )

    @property
    def trace_id(self) -> str | None:
        return self._observation.trace_id

    @property
    def span_id(self) -> str | None:
        return self._observation.id

    def _update(self, attributes: Mapping[str, AttributeValue]) -> None:
        fields = update_fields(self._as_type, attributes)
        if "usage_details" in fields:
            self._usage.update(fields["usage_details"])
            fields["usage_details"] = dict(self._usage)
        if fields:
            self._observation.update(**fields)


_current: ContextVar[LangfuseSpan | None] = ContextVar(
    "langfuse_current_span", default=None
)


class LangfuseTracer(BaseTracer):
    """Emits spans as Langfuse observations through the official SDK.

    Without `client`, it needs `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`
    in the environment (`LANGFUSE_BASE_URL` selects the server); when either
    is unset it behaves like `NoOpTracer` and never builds a client.

    Requires the `agent_platform[tracing]` extra.
    """

    def __init__(self, client: Langfuse | None = None) -> None:
        self._client: Langfuse | None = client
        self._noop = NoOpTracer()
        if client is None:
            if all(os.environ.get(name) for name in _REQUIRED_ENV):
                self._client = Langfuse()
            else:
                logger.info(
                    "Langfuse tracing is off: set %s and %s to enable it.",
                    *_REQUIRED_ENV,
                )

    @contextmanager
    def start_span(
        self, name: str, attributes: Mapping[str, AttributeValue] | None = None
    ) -> Iterator[Span]:
        client = self._client
        if client is None:
            with self._noop.start_span(name, attributes) as span:
                yield span
            return

        start = attributes or {}
        as_type = observation_type(start)
        fields = update_fields(as_type, start)
        with client.start_as_current_observation(
            name=name, as_type=as_type, **fields
        ) as observation:
            span = LangfuseSpan(observation, as_type, fields.get("usage_details"))
            token = _current.set(span)
            try:
                yield span
            finally:
                _current.reset(token)

    def current_span(self) -> Span:
        if self._client is None:
            return self._noop.current_span()
        return _current.get() or _NOOP_SPAN
