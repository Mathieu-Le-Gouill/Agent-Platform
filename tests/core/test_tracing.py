import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.trace import StatusCode

from agent_platform.core import tracing
from agent_platform.core.tracing import (
    TracingBackend,
    TracingConfig,
    _otlp_endpoint_configured,
    _resolve_exporter,
    configure_tracing,
    mark_span_error,
    traced_span,
)


class TestTracingConfigFromEnv:
    def test_defaults_to_auto(self, monkeypatch):
        monkeypatch.delenv("AGENT_PLATFORM_TRACING", raising=False)
        assert TracingConfig.from_env().backend == TracingBackend.AUTO

    def test_parses_none(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_TRACING", "none")
        assert TracingConfig.from_env().backend == TracingBackend.NONE

    def test_parses_console(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_TRACING", "CONSOLE")
        assert TracingConfig.from_env().backend == TracingBackend.CONSOLE

    def test_invalid_value_defaults_to_auto(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_TRACING", "not-a-backend")
        assert TracingConfig.from_env().backend == TracingBackend.AUTO

    def test_reads_service_name(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_SERVICE_NAME", "my-service")
        assert TracingConfig.from_env().service_name == "my-service"

    def test_falls_back_to_otel_service_name(self, monkeypatch):
        monkeypatch.delenv("AGENT_PLATFORM_SERVICE_NAME", raising=False)
        monkeypatch.setenv("OTEL_SERVICE_NAME", "otel-service")
        assert TracingConfig.from_env().service_name == "otel-service"


class TestOtlpEndpointConfigured:
    def test_false_when_unset(self):
        assert _otlp_endpoint_configured() is False

    def test_true_when_endpoint_set(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        assert _otlp_endpoint_configured() is True

    def test_true_when_traces_endpoint_set(self, monkeypatch):
        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4318/v1/traces"
        )
        assert _otlp_endpoint_configured() is True


class TestResolveExporter:
    def test_none_backend_returns_none(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        assert _resolve_exporter(TracingConfig(backend=TracingBackend.NONE)) is None

    def test_console_backend_returns_console_exporter(self):
        exporter = _resolve_exporter(TracingConfig(backend=TracingBackend.CONSOLE))
        assert isinstance(exporter, ConsoleSpanExporter)

    def test_auto_without_endpoint_is_a_safe_noop(self):
        assert _resolve_exporter(TracingConfig(backend=TracingBackend.AUTO)) is None

    def test_auto_with_endpoint_returns_otlp_http_exporter(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")

        exporter = _resolve_exporter(TracingConfig(backend=TracingBackend.AUTO))

        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        assert isinstance(exporter, OTLPSpanExporter)

    def test_auto_honors_grpc_protocol_env_var(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")

        exporter = _resolve_exporter(TracingConfig(backend=TracingBackend.AUTO))

        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPGrpcSpanExporter,
        )

        assert isinstance(exporter, OTLPGrpcSpanExporter)


class TestConfigureTracing:
    def test_auto_without_endpoint_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(tracing, "_configured", False)
        configure_tracing(TracingConfig(backend=TracingBackend.AUTO))
        assert tracing._configured is False

    def test_none_backend_is_a_noop_even_with_endpoint_set(self, monkeypatch):
        monkeypatch.setattr(tracing, "_configured", False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        configure_tracing(TracingConfig(backend=TracingBackend.NONE))
        assert tracing._configured is False

    def test_console_backend_installs_a_real_provider(self, isolated_global_provider):
        configure_tracing(TracingConfig(backend=TracingBackend.CONSOLE))

        assert tracing._configured is True
        assert not isinstance(
            trace_api.get_tracer_provider(), trace_api.ProxyTracerProvider
        )

    def test_defers_to_an_already_configured_provider(self, isolated_global_provider):
        from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

        preexisting = SDKTracerProvider()
        trace_api.set_tracer_provider(preexisting)

        configure_tracing(TracingConfig(backend=TracingBackend.CONSOLE))

        assert trace_api.get_tracer_provider() is preexisting

    def test_exporter_override_bypasses_backend_resolution(
        self, isolated_global_provider
    ):
        exporter = ConsoleSpanExporter()
        configure_tracing(TracingConfig(backend=TracingBackend.NONE), exporter=exporter)
        assert tracing._configured is True


class TestTracedSpan:
    def test_records_name_and_attributes(self, recorded_spans):
        with traced_span("execute_tool", {"gen_ai.tool.name": "search"}):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "execute_tool"
        assert span.attributes["gen_ai.tool.name"] == "search"
        assert span.status.status_code == StatusCode.UNSET

    def test_skips_none_attributes(self, recorded_spans):
        with traced_span("chat", {"gen_ai.request.model": None}):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert "gen_ai.request.model" not in span.attributes

    def test_records_exception_and_reraises(self, recorded_spans):
        with pytest.raises(ValueError):
            with traced_span("chat"):
                raise ValueError("boom")

        (span,) = recorded_spans.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.events[0].name == "exception"


class TestMarkSpanError:
    def test_records_exception_by_default(self, recorded_spans):
        with traced_span("chat") as span:
            mark_span_error(span, ValueError("boom"))

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["error.type"] == "ValueError"
        assert span.status.status_code == StatusCode.ERROR
        assert span.events[0].name == "exception"

    def test_skips_record_exception_when_disabled(self, recorded_spans):
        with traced_span("execute_tool") as span:
            mark_span_error(span, ValueError("boom"), record_exception=False)

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["error.type"] == "ValueError"
        assert span.status.status_code == StatusCode.ERROR
        assert len(span.events) == 0
