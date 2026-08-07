from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agent_platform.integrations.tracing.otel.config import (
    OTelBackend,
    OTelTracingConfig,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter

__all__ = ["otlp_endpoint_configured", "resolve_exporter"]


def otlp_endpoint_configured() -> bool:
    return bool(
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )


def build_otlp_exporter() -> SpanExporter:
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


def resolve_exporter(config: OTelTracingConfig) -> SpanExporter | None:
    """Pick the span exporter `config.backend` asks for, or `None` for no export."""
    if config.backend is OTelBackend.NONE:
        return None

    if config.backend is OTelBackend.CONSOLE:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter()

    # AUTO: only export when the standard env vars actually point somewhere;
    # otherwise stay a safe no-op instead of falling back to the OTLP
    # exporter's own default of localhost:4318.
    if not otlp_endpoint_configured():
        return None
    return build_otlp_exporter()
