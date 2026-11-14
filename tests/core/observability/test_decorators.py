import pytest

from agent_platform.core.observability import (
    ComponentRun,
    PipelineRun,
    current,
    traced,
)


class TestTraced:
    async def test_traces_an_async_function(self, recorded_spans):
        @traced(PipelineRun, name="rag.query")
        async def run() -> str:
            return "done"

        assert await run() == "done"

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "pipeline"
        assert span.attributes["pipeline.name"] == "rag.query"

    def test_traces_a_sync_function(self, recorded_spans):
        @traced(ComponentRun, name="chunker")
        def run() -> str:
            return "done"

        assert run() == "done"

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "chunker"
        assert span.attributes["operation.name"] == "chunker"

    def test_keeps_the_wrapped_function_identity(self):
        @traced(PipelineRun, name="rag.query")
        async def run() -> None:
            """Docstring."""

        assert run.__name__ == "run"
        assert run.__doc__ == "Docstring."

    def test_builds_a_fresh_observation_per_call(self):
        seen = []

        @traced(PipelineRun, name="rag.query")
        def run() -> None:
            seen.append(current(PipelineRun))

        run()
        run()

        first, second = seen
        assert first is not second

    def test_propagates_exceptions(self, recorded_spans):
        @traced(PipelineRun, name="rag.query")
        def run() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            run()

        assert len(recorded_spans.get_finished_spans()) == 1
