import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util._once import Once

from agent_platform.core import tracing
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.tracing import (
    GenAIAttributes,
    TracingBackend,
    TracingConfig,
    _otlp_endpoint_configured,
    _resolve_exporter,
    configure_tracing,
    record_token_usage,
    traced_operation_span,
    traced_span,
)


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
        with traced_span("execute_tool", **{GenAIAttributes.TOOL_NAME: "search"}):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "execute_tool"
        assert span.attributes[GenAIAttributes.TOOL_NAME] == "search"
        assert span.status.status_code == StatusCode.UNSET

    def test_skips_none_attributes(self, recorded_spans):
        with traced_span("chat", **{GenAIAttributes.REQUEST_MODEL: None}):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert GenAIAttributes.REQUEST_MODEL not in span.attributes

    def test_records_exception_and_reraises(self, recorded_spans):
        with pytest.raises(ValueError):
            with traced_span("chat"):
                raise ValueError("boom")

        (span,) = recorded_spans.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.events[0].name == "exception"


class TestRecordTokenUsage:
    def test_sets_usage_attributes(self, recorded_spans):
        with traced_span("chat") as span:
            record_token_usage(span, TokenUsage(input_tokens=12, output_tokens=34))

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes[GenAIAttributes.USAGE_INPUT_TOKENS] == 12
        assert span.attributes[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 34


class TestTracedOperationSpan:
    def test_names_span_after_operation(self, recorded_spans):
        with traced_operation_span("chat"):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "chat"

    def test_sets_operation_name_attribute(self, recorded_spans):
        with traced_operation_span("execute_tool"):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes[GenAIAttributes.OPERATION_NAME] == "execute_tool"

    def test_merges_extra_attributes(self, recorded_spans):
        with traced_operation_span("chat", **{GenAIAttributes.REQUEST_MODEL: "gpt-4"}):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes[GenAIAttributes.OPERATION_NAME] == "chat"
        assert span.attributes[GenAIAttributes.REQUEST_MODEL] == "gpt-4"
