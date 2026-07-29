from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from opentelemetry import trace
from opentelemetry.trace import ProxyTracerProvider, Span, Status, StatusCode
from pydantic import BaseModel

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter

__all__ = [
    "TracingBackend",
    "TracingConfig",
    "configure_tracing",
    "get_tracer",
    "mark_span_error",
    "traced_span",
]

_TRACER_NAME = "agent_platform"
_configure_lock = threading.Lock()
_configured = False
_ERROR_TYPE_ATTR = "error.type"


class TracingBackend(StrEnum):
    AUTO = "auto"
    NONE = "none"
    CONSOLE = "console"


class TracingConfig(BaseModel):
    backend: TracingBackend = TracingBackend.AUTO
    service_name: str = "agent-platform"

    @classmethod
    def from_env(cls) -> TracingConfig:
        raw = os.getenv("AGENT_PLATFORM_TRACING", "auto").lower()
        try:
            backend = TracingBackend(raw)
        except ValueError:
            backend = TracingBackend.AUTO
        service_name = os.getenv(
            "AGENT_PLATFORM_SERVICE_NAME",
            os.getenv("OTEL_SERVICE_NAME", "agent-platform"),
        )
        return cls(backend=backend, service_name=service_name)


def _otlp_endpoint_configured() -> bool:
    return bool(
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )


def _build_otlp_exporter() -> SpanExporter:
    """Build an OTLP span exporter from the standard `OTEL_EXPORTER_OTLP_*` env vars.

    This is how every OTel SDK, in every language, is pointed at a backend:
    `OTEL_EXPORTER_OTLP_ENDPOINT` (+ `_HEADERS` for auth) is understood by
    every OTLP-speaking backend (Honeycomb, Grafana Tempo/Cloud, Datadog,
    New Relic, SigNoz, a self-hosted OTel Collector fanning out to
    Jaeger/Zipkin/X-Ray, LangSmith, Langfuse, ...), so there is nothing
    agent-platform-specific to configure here beyond the transport.
    `OTEL_EXPORTER_OTLP_PROTOCOL` picks `http/protobuf` (default) or `grpc`.
    """
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf").lower()

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPGrpcSpanExporter,
        )

        return OTLPGrpcSpanExporter()

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPHttpSpanExporter,
    )

    return OTLPHttpSpanExporter()


def _resolve_exporter(config: TracingConfig) -> SpanExporter | None:
    if config.backend is TracingBackend.NONE:
        return None

    if config.backend is TracingBackend.CONSOLE:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()

    # AUTO: only export when the standard env vars actually point somewhere;
    # otherwise stay a safe no-op instead of falling back to the OTLP
    # exporter's own default of localhost:4318.
    if not _otlp_endpoint_configured():
        return None
    return _build_otlp_exporter()


def configure_tracing(
    config: TracingConfig | None = None, *, exporter: SpanExporter | None = None
) -> None:
    """Point the global OpenTelemetry tracer provider at `config.backend`.

    Call once at process startup. With the default `AUTO` backend, this is a
    no-op unless `OTEL_EXPORTER_OTLP_ENDPOINT` (or `_TRACES_ENDPOINT`) is set,
    so it's always safe to call unconditionally - no separate "enable
    tracing" flag is needed beyond pointing the standard env var at a
    backend. `AGENT_PLATFORM_TRACING` only needs to be set to override that
    default: `none` force-disables even if an OTLP endpoint env var happens
    to be set, `console` forces local stdout export for debugging.

    If a tracer provider has already been installed by something else (the
    `opentelemetry-instrument` auto-instrumentation wrapper, an APM agent,
    or the host application embedding agent_platform), this defers to it
    instead of overwriting it, so spans still end up wherever that provider
    sends them.

    `exporter` is an escape hatch for backends with no standard OTLP
    ingestion path: pass a pre-built `SpanExporter` and it is used as-is,
    bypassing `config.backend` entirely.

    Requires the `agent_platform[tracing]` extra; without it, `traced_span`
    still works but spans are dropped by OpenTelemetry's default no-op
    provider.
    """
    global _configured
    config = config or TracingConfig.from_env()

    with _configure_lock:
        if _configured:
            return

        if not isinstance(trace.get_tracer_provider(), ProxyTracerProvider):
            # A real provider is already installed; don't fight it.
            _configured = True
            return

        resolved_exporter = (
            exporter if exporter is not None else _resolve_exporter(config)
        )
        if resolved_exporter is None:
            return

        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )

        provider = TracerProvider(
            resource=Resource.create({"service.name": config.service_name})
        )
        processor = (
            SimpleSpanProcessor(resolved_exporter)
            if config.backend is TracingBackend.CONSOLE
            else BatchSpanProcessor(resolved_exporter)
        )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        _configured = True


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def traced_span(
    name: str, /, attributes: Mapping[str, Any] | None = None
) -> Iterator[Span]:
    """Start a span named `name` with `attributes`, recording latency and errors.

    Safe to use unconditionally: when tracing hasn't been configured (or the
    `tracing` extra isn't installed), OpenTelemetry's default tracer is a
    no-op, so this becomes a cheap pass-through.
    """
    with get_tracer().start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            mark_span_error(span, exc)
            raise


def mark_span_error(
    span: Span, exc: Exception, *, record_exception: bool = True
) -> None:
    """Tag `span` as failed using OTel's error-status conventions.

    Shared by `traced_span`'s automatic handling of propagating exceptions
    and call sites that catch an error without re-raising (e.g. tool calls
    swallowed into a `ToolResult`), which pass `record_exception=False`
    since the exception object itself never leaves that call site.
    """
    span.set_attribute(_ERROR_TYPE_ATTR, type(exc).__qualname__)
    if record_exception:
        span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
