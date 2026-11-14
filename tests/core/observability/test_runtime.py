from dataclasses import dataclass
from typing import Annotated

import pytest

from agent_platform.core.interfaces.tracing.noop import NoOpSpan, NoOpTracer
from agent_platform.core.observability import (
    Fact,
    Observation,
    current,
    emit,
    get_tracer,
    observe,
    reset_tracer,
    set_tracer,
)


@dataclass
class Job(Observation):
    operation = "job"

    kind: Annotated[str, Fact("job.kind")]
    result: Annotated[str | None, Fact("job.result")] = None


@dataclass
class Note(Observation):
    """An event: no operation of its own, so it lands on the enclosing span."""

    text: Annotated[str, Fact("note.text")]


class RecordingTracer(NoOpTracer):
    """Records what `observe` forwards, without needing a real backend."""

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
        with observe(Job(kind="index")) as job:
            job.result = "ok"

        assert job.result == "ok"

    def test_still_propagates_exceptions(self):
        with pytest.raises(ValueError):
            with observe(Job(kind="index")):
                raise ValueError("boom")

    def test_noop_span_satisfies_the_span_protocol(self):
        from agent_platform.core.interfaces.tracing.span import Span

        assert isinstance(NoOpSpan(), Span)


class TestObserveSpans:
    def test_names_the_span_after_the_operation(self, recorded_spans):
        with observe(Job(kind="index")):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "job"
        assert span.error is None

    def test_records_what_was_known_when_the_span_opened(self):
        tracer = RecordingTracer()
        set_tracer(tracer)

        with observe(Job(kind="index")):
            pass

        (name, attributes) = tracer.calls[0]
        assert name == "job"
        assert attributes["job.kind"] == "index"
        assert "job.result" not in attributes

    def test_flushes_what_changed_before_the_span_ends(self, recorded_spans):
        with observe(Job(kind="index")) as job:
            job.result = "ok"

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["job.result"] == "ok"

    def test_records_a_raised_exception_and_reraises(self, recorded_spans):
        with pytest.raises(ValueError):
            with observe(Job(kind="index")):
                raise ValueError("boom")

        (span,) = recorded_spans.get_finished_spans()
        assert span.error is not None
        assert span.record_exception is True

    def test_records_a_swallowed_error_without_a_stack_trace(self, recorded_spans):
        with observe(Job(kind="index")) as job:
            job.record_error(ValueError("boom"))

        (span,) = recorded_spans.get_finished_spans()
        assert span.error is not None
        assert span.record_exception is False

    def test_a_raised_error_is_recorded_once_even_if_also_stored(self, recorded_spans):
        with pytest.raises(ValueError):
            with observe(Job(kind="index")) as job:
                job.record_error(RuntimeError("swallowed"))
                raise ValueError("boom")

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["error.type"] == "ValueError"

    def test_nests_spans_under_the_active_one(self, recorded_spans):
        with observe(Job(kind="outer")):
            with observe(Job(kind="inner")):
                pass

        child, parent = recorded_spans.get_finished_spans()
        assert child.parent is parent


class TestEmit:
    def test_stamps_facts_on_the_enclosing_span(self, recorded_spans):
        with observe(Job(kind="index")):
            emit(Note(text="hello"))

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["note.text"] == "hello"

    def test_is_a_noop_outside_any_span(self, recorded_spans):
        emit(Note(text="hello"))

        assert recorded_spans.get_finished_spans() == ()


class TestCurrent:
    def test_finds_the_enclosing_observation(self):
        with observe(Job(kind="index")) as job:
            assert current(Job) is job

    def test_finds_the_innermost_one(self):
        with observe(Job(kind="outer")):
            with observe(Job(kind="inner")) as inner:
                assert current(Job) is inner

    def test_returns_none_when_no_such_observation_is_open(self):
        assert current(Job) is None
        with observe(Job(kind="index")):
            assert current(Note) is None

    def test_the_stack_unwinds_on_failure(self):
        with pytest.raises(ValueError):
            with observe(Job(kind="index")):
                raise ValueError("boom")

        assert current(Job) is None
