import pytest
from opentelemetry.trace import StatusCode

from agent_platform.core.interfaces.tracing.noop import NoOpSpan, NoOpTracer
from agent_platform.core.tracing import (
    get_tracer,
    reset_tracer,
    set_tracer,
    traced_span,
)


class RecordingTracer(NoOpTracer):
    """Records what `traced_span` forwards, without needing a real backend."""

    def __init__(self):
        self.calls = []

    def start_span(self, name, attributes=None):
        self.calls.append((name, dict(attributes or {})))
        return super().start_span(name, attributes)


@pytest.fixture(autouse=True)
def restore_default_tracer():
    yield
    reset_tracer()


class TestTracerRegistry:
    def test_defaults_to_noop(self):
        reset_tracer()
        assert isinstance(get_tracer(), NoOpTracer)

    def test_set_tracer_installs_it(self):
        tracer = NoOpTracer()
        set_tracer(tracer)
        assert get_tracer() is tracer

    def test_reset_restores_noop(self):
        set_tracer(NoOpTracer())
        installed = get_tracer()
        reset_tracer()
        assert get_tracer() is not installed


class TestNoOpTracer:
    def test_yields_a_span_that_swallows_everything(self):
        with traced_span("chat", {"gen_ai.request.model": "gpt-4"}) as span:
            span.set_attribute("any", "value")
            span.record_error(ValueError("boom"), record_exception=False)
            assert span.trace_id is None
            assert span.span_id is None

    def test_still_propagates_exceptions(self):
        with pytest.raises(ValueError):
            with traced_span("chat"):
                raise ValueError("boom")

    def test_noop_span_satisfies_the_span_protocol(self):
        from agent_platform.core.interfaces.tracing.span import Span

        assert isinstance(NoOpSpan(), Span)


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

    def test_nests_spans_under_the_active_one(self, recorded_spans):
        with traced_span("invoke_agent"):
            with traced_span("execute_tool"):
                pass

        child, parent = recorded_spans.get_finished_spans()
        assert child.name == "execute_tool"
        assert child.parent.span_id == parent.context.span_id

    def test_exposes_trace_and_span_ids(self, recorded_spans):
        with traced_span("chat") as span:
            trace_id, span_id = span.trace_id, span.span_id

        (finished,) = recorded_spans.get_finished_spans()
        assert trace_id == format(finished.context.trace_id, "032x")
        assert span_id == format(finished.context.span_id, "016x")

    def test_delegates_to_the_installed_tracer(self):
        tracer = RecordingTracer()
        set_tracer(tracer)

        with traced_span("chat", {"a": 1, "b": None}):
            pass

        assert tracer.calls == [("chat", {"a": 1})]
