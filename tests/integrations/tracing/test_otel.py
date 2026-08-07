import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agent_platform.core.interfaces.tracing.base import BaseTracer
from agent_platform.core.interfaces.tracing.span import Span
from agent_platform.integrations.tracing.otel.config import (
    OTelBackend,
    OTelTracingConfig,
)
from agent_platform.integrations.tracing.otel.exporters import (
    otlp_endpoint_configured,
    resolve_exporter,
)
from agent_platform.integrations.tracing.otel.provider import OTelSpan, OTelTracer


@pytest.fixture
def recording_tracer():
    """An `OTelTracer` writing to an in-memory exporter, off the global provider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OTelTracer(tracer_provider=provider), exporter


class TestOTelTracingConfigFromEnv:
    def test_defaults_to_auto(self, monkeypatch):
        monkeypatch.delenv("AGENT_PLATFORM_TRACING", raising=False)
        assert OTelTracingConfig.from_env().backend == OTelBackend.AUTO

    def test_parses_none(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_TRACING", "none")
        assert OTelTracingConfig.from_env().backend == OTelBackend.NONE

    def test_parses_console(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_TRACING", "CONSOLE")
        assert OTelTracingConfig.from_env().backend == OTelBackend.CONSOLE

    def test_invalid_value_defaults_to_auto(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_TRACING", "not-a-backend")
        assert OTelTracingConfig.from_env().backend == OTelBackend.AUTO

    def test_reads_service_name(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_SERVICE_NAME", "my-service")
        assert OTelTracingConfig.from_env().service_name == "my-service"

    def test_falls_back_to_otel_service_name(self, monkeypatch):
        monkeypatch.delenv("AGENT_PLATFORM_SERVICE_NAME", raising=False)
        monkeypatch.setenv("OTEL_SERVICE_NAME", "otel-service")
        assert OTelTracingConfig.from_env().service_name == "otel-service"


class TestOtlpEndpointConfigured:
    def test_false_when_unset(self):
        assert otlp_endpoint_configured() is False

    def test_true_when_endpoint_set(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        assert otlp_endpoint_configured() is True

    def test_true_when_traces_endpoint_set(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4318/v1/traces"
        )
        assert otlp_endpoint_configured() is True


class TestResolveExporter:
    def test_none_backend_returns_none(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        assert resolve_exporter(OTelTracingConfig(backend=OTelBackend.NONE)) is None

    def test_console_backend_returns_console_exporter(self):
        exporter = resolve_exporter(OTelTracingConfig(backend=OTelBackend.CONSOLE))
        assert isinstance(exporter, ConsoleSpanExporter)

    def test_auto_without_endpoint_is_a_safe_noop(self):
        assert resolve_exporter(OTelTracingConfig(backend=OTelBackend.AUTO)) is None

    def test_auto_with_endpoint_returns_otlp_http_exporter(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")

        exporter = resolve_exporter(OTelTracingConfig(backend=OTelBackend.AUTO))

        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        assert isinstance(exporter, OTLPSpanExporter)

    def test_auto_honors_grpc_protocol_env_var(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")

        exporter = resolve_exporter(OTelTracingConfig(backend=OTelBackend.AUTO))

        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPGrpcSpanExporter,
        )

        assert isinstance(exporter, OTLPGrpcSpanExporter)


class TestGlobalProviderInstallation:
    def test_auto_without_endpoint_installs_nothing(self, isolated_global_provider):
        OTelTracer(OTelTracingConfig(backend=OTelBackend.AUTO))
        assert isinstance(
            trace_api.get_tracer_provider(), trace_api.ProxyTracerProvider
        )

    def test_none_backend_installs_nothing_even_with_endpoint_set(
        self, isolated_global_provider, monkeypatch
    ):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        OTelTracer(OTelTracingConfig(backend=OTelBackend.NONE))
        assert isinstance(
            trace_api.get_tracer_provider(), trace_api.ProxyTracerProvider
        )

    def test_console_backend_installs_a_real_provider(self, isolated_global_provider):
        OTelTracer(OTelTracingConfig(backend=OTelBackend.CONSOLE))
        assert not isinstance(
            trace_api.get_tracer_provider(), trace_api.ProxyTracerProvider
        )

    def test_defers_to_an_already_configured_provider(self, isolated_global_provider):
        preexisting = TracerProvider()
        trace_api.set_tracer_provider(preexisting)

        OTelTracer(OTelTracingConfig(backend=OTelBackend.CONSOLE))

        assert trace_api.get_tracer_provider() is preexisting

    def test_exporter_override_bypasses_backend_resolution(
        self, isolated_global_provider
    ):
        OTelTracer(
            OTelTracingConfig(backend=OTelBackend.NONE), exporter=ConsoleSpanExporter()
        )
        assert not isinstance(
            trace_api.get_tracer_provider(), trace_api.ProxyTracerProvider
        )

    def test_explicit_provider_leaves_the_global_one_alone(
        self, isolated_global_provider
    ):
        OTelTracer(tracer_provider=TracerProvider())
        assert isinstance(
            trace_api.get_tracer_provider(), trace_api.ProxyTracerProvider
        )


class TestOTelTracer:
    def test_implements_the_base_tracer_interface(self, recording_tracer):
        tracer, _ = recording_tracer
        assert isinstance(tracer, BaseTracer)

    def test_emits_named_span_with_attributes(self, recording_tracer):
        tracer, exporter = recording_tracer

        with tracer.start_span("chat", {"gen_ai.request.model": "gpt-4"}):
            pass

        (span,) = exporter.get_finished_spans()
        assert span.name == "chat"
        assert span.attributes["gen_ai.request.model"] == "gpt-4"

    def test_yields_a_span_satisfying_the_protocol(self, recording_tracer):
        tracer, _ = recording_tracer

        with tracer.start_span("chat") as span:
            assert isinstance(span, OTelSpan)
            assert isinstance(span, Span)

    def test_makes_the_span_current_so_children_nest(self, recording_tracer):
        tracer, exporter = recording_tracer

        with tracer.start_span("invoke_agent"):
            with tracer.start_span("execute_tool"):
                pass

        child, parent = exporter.get_finished_spans()
        assert child.parent.span_id == parent.context.span_id


class TestOTelSpan:
    def test_set_attribute(self, recording_tracer):
        tracer, exporter = recording_tracer

        with tracer.start_span("chat") as span:
            span.set_attribute("gen_ai.usage.input_tokens", 12)

        (span,) = exporter.get_finished_spans()
        assert span.attributes["gen_ai.usage.input_tokens"] == 12

    def test_record_error_records_exception_by_default(self, recording_tracer):
        tracer, exporter = recording_tracer

        with tracer.start_span("chat") as span:
            span.record_error(ValueError("boom"))

        (span,) = exporter.get_finished_spans()
        assert span.attributes["error.type"] == "ValueError"
        assert span.status.status_code == StatusCode.ERROR
        assert span.events[0].name == "exception"

    def test_record_error_skips_exception_when_disabled(self, recording_tracer):
        tracer, exporter = recording_tracer

        with tracer.start_span("execute_tool") as span:
            span.record_error(ValueError("boom"), record_exception=False)

        (span,) = exporter.get_finished_spans()
        assert span.attributes["error.type"] == "ValueError"
        assert span.status.status_code == StatusCode.ERROR
        assert len(span.events) == 0

    def test_exposes_hex_trace_and_span_ids(self, recording_tracer):
        tracer, exporter = recording_tracer

        with tracer.start_span("chat") as span:
            trace_id, span_id = span.trace_id, span.span_id

        (finished,) = exporter.get_finished_spans()
        assert trace_id == format(finished.context.trace_id, "032x")
        assert span_id == format(finished.context.span_id, "016x")

    def test_ids_are_none_for_an_invalid_context(self):
        span = OTelSpan(trace_api.INVALID_SPAN)
        assert span.trace_id is None
        assert span.span_id is None
