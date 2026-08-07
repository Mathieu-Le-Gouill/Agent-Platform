from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.trace import (
    ProxyTracerProvider,
    Status,
    StatusCode,
    format_span_id,
    format_trace_id,
)
from opentelemetry.trace import Span as OTelSpanType

from agent_platform.core.interfaces.tracing.base import BaseTracer
from agent_platform.core.interfaces.tracing.span import AttributeValue, Span
from agent_platform.integrations.tracing.otel.config import (
    OTelBackend,
    OTelTracingConfig,
)
from agent_platform.integrations.tracing.otel.exporters import resolve_exporter

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter
    from opentelemetry.trace import TracerProvider

__all__ = ["OTelSpan", "OTelTracer"]

_TRACER_NAME = "agent_platform"
_ERROR_TYPE_ATTR = "error.type"
_install_lock = threading.Lock()


class OTelSpan:
    """Adapts an OpenTelemetry span to the platform's `Span` protocol."""

    __slots__ = ("_span",)

    def __init__(self, span: OTelSpanType) -> None:
        self._span = span

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        self._span.set_attribute(key, value)

    def record_error(self, exc: Exception, *, record_exception: bool = True) -> None:
        self._span.set_attribute(_ERROR_TYPE_ATTR, type(exc).__qualname__)
        if record_exception:
            self._span.record_exception(exc)
        self._span.set_status(Status(StatusCode.ERROR, str(exc)))

    @property
    def trace_id(self) -> str | None:
        context = self._span.get_span_context()
        return format_trace_id(context.trace_id) if context.is_valid else None

    @property
    def span_id(self) -> str | None:
        context = self._span.get_span_context()
        return format_span_id(context.span_id) if context.is_valid else None


class OTelTracer(BaseTracer):
    """Emits spans through OpenTelemetry, exporting over OTLP by default.

    Any OTLP-speaking backend works without code changes (Langfuse, LangSmith,
    Phoenix, Honeycomb, Grafana Tempo, Datadog, an OTel Collector, ...); see
    `integrations/README.md` for per-vendor endpoint and auth env vars.

    Constructing this installs a global `TracerProvider` unless one is already
    present, so it is the object that owns process-global OTel state; `core`
    only ever sees the `BaseTracer` interface.

    Requires the `agent_platform[tracing]` extra.
    """

    def __init__(
        self,
        config: OTelTracingConfig | None = None,
        *,
        exporter: SpanExporter | None = None,
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        """Wire up OpenTelemetry.

        `exporter` is an escape hatch for backends with no standard OTLP
        ingestion path: a pre-built `SpanExporter` is used as-is, bypassing
        `config.backend`. `tracer_provider` bypasses global installation
        entirely and emits into the given provider, for tests and for hosts
        that manage their own provider.
        """
        self._config = config or OTelTracingConfig.from_env()

        if tracer_provider is not None:
            self._tracer = tracer_provider.get_tracer(_TRACER_NAME)
            return

        self._install_global_provider(exporter)
        self._tracer = trace.get_tracer(_TRACER_NAME)

    def _install_global_provider(self, exporter: SpanExporter | None) -> None:
        with _install_lock:
            if not isinstance(trace.get_tracer_provider(), ProxyTracerProvider):
                # A real provider is already installed by the
                # `opentelemetry-instrument` wrapper, an APM agent, or the
                # host application. Don't fight it: spans still reach
                # wherever that provider sends them.
                return

            resolved = (
                exporter if exporter is not None else resolve_exporter(self._config)
            )
            if resolved is None:
                # Nothing to export to. Leave the no-op proxy provider in
                # place so spans are created and dropped cheaply.
                return

            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                SimpleSpanProcessor,
            )

            provider = SDKTracerProvider(
                resource=Resource.create({"service.name": self._config.service_name})
            )
            processor = (
                SimpleSpanProcessor(resolved)
                if self._config.backend is OTelBackend.CONSOLE
                else BatchSpanProcessor(resolved)
            )
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)

    @contextmanager
    def start_span(
        self, name: str, attributes: Mapping[str, AttributeValue] | None = None
    ) -> Iterator[Span]:
        with self._tracer.start_as_current_span(
            name, attributes=dict(attributes) if attributes else None
        ) as span:
            yield OTelSpan(span)
