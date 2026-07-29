import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util._once import Once

from agent_platform.core import tracing


@pytest.fixture
def recorded_spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(tracing, "get_tracer", lambda: tracer)
    return exporter


@pytest.fixture
def isolated_global_provider(monkeypatch):
    """Reset OpenTelemetry's process-global tracer provider for one test.

    `trace.set_tracer_provider` only ever takes effect once per process, so
    tests that actually install a provider (rather than monkeypatching
    `get_tracer` directly) need their own clean slate.
    """
    monkeypatch.setattr(trace_api, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(trace_api, "_TRACER_PROVIDER_SET_ONCE", Once())
    monkeypatch.setattr(tracing, "_configured", False)


@pytest.fixture(autouse=True)
def clear_otlp_env(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
