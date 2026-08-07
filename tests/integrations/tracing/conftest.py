import opentelemetry.trace as trace_api
import pytest
from opentelemetry.util._once import Once


@pytest.fixture
def isolated_global_provider(monkeypatch):
    """Reset OpenTelemetry's process-global tracer provider for one test.

    `trace.set_tracer_provider` only ever takes effect once per process, so
    tests that actually install a provider (rather than passing
    `tracer_provider=` explicitly) need their own clean slate.
    """
    monkeypatch.setattr(trace_api, "_TRACER_PROVIDER", None)
    monkeypatch.setattr(trace_api, "_TRACER_PROVIDER_SET_ONCE", Once())


@pytest.fixture(autouse=True)
def clear_otlp_env(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
