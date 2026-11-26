import asyncio
import json
import logging
from dataclasses import dataclass
from itertools import count
from typing import Annotated

import pytest
from langfuse import Langfuse
from langfuse._client.attributes import LangfuseOtelSpanAttributes as Attr
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from agent_platform.core.interfaces.tracing.noop import NoOpSpan
from agent_platform.core.observability import (
    AgentRun,
    Fact,
    GuardrailCheck,
    LLMCall,
    ModelCost,
    Observation,
    PipelineRun,
    Retrieval,
    ToolRun,
    current_span,
    emit,
    observe,
    reset_tracer,
    set_tracer,
)
from agent_platform.integrations.tracing.langfuse.provider import LangfuseTracer

_keys = count()


@dataclass
class Other(Observation):
    operation = "something_else"

    label: Annotated[str, Fact("other.label")]


class Exported:
    """The spans a test exported, queryable by name."""

    def __init__(self, exporter: InMemorySpanExporter, client: Langfuse) -> None:
        self._exporter = exporter
        self._client = client

    def spans(self):
        self._client.flush()
        return self._exporter.get_finished_spans()

    def get(self, name: str):
        (span,) = [s for s in self.spans() if s.name == name]
        return span


@pytest.fixture
def exported():
    # The SDK keeps one resource manager per public key, so each test needs
    # its own key to get its own in-memory exporter.
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=f"pk-test-{next(_keys)}",
        secret_key="sk-test",
        base_url="http://localhost:1",
        tracer_provider=TracerProvider(),
        span_exporter=exporter,
    )
    set_tracer(LangfuseTracer(client))
    yield Exported(exporter, client)
    reset_tracer()


def metadata(span, key: str):
    return json.loads(span.attributes[f"{Attr.OBSERVATION_METADATA}.{key}"])


def kind(span) -> str:
    return span.attributes[Attr.OBSERVATION_TYPE]


class TestObservationTypes:
    def test_an_llm_call_is_a_generation_with_model_usage_and_metadata(self, exported):
        with observe(LLMCall(provider="openai", request_model="gpt-4o-mini")) as call:
            call.response_model = "gpt-4o-mini-2024"
            call.input_tokens = 12
            call.output_tokens = 34

        span = exported.get("chat")
        assert kind(span) == "generation"
        assert span.attributes[Attr.OBSERVATION_MODEL] == "gpt-4o-mini-2024"
        assert json.loads(span.attributes[Attr.OBSERVATION_USAGE_DETAILS]) == {
            "input": 12,
            "output": 34,
        }
        assert metadata(span, "llm.provider") == "openai"
        assert metadata(span, "operation.name") == "chat"

    def test_the_request_model_is_used_when_no_response_model_arrives(self, exported):
        with observe(LLMCall(provider="openai", request_model="gpt-4o-mini")):
            pass

        span = exported.get("chat")
        assert span.attributes[Attr.OBSERVATION_MODEL] == "gpt-4o-mini"

    def test_usage_arriving_in_two_calls_ends_as_one_usage_details(self, exported):
        with observe(LLMCall(provider="openai", request_model="m")):
            tracer_span = current_span()
            tracer_span.set_attribute("llm.usage.input_tokens", 5)
            tracer_span.set_attribute("llm.usage.output_tokens", 7)

        span = exported.get("chat")
        assert json.loads(span.attributes[Attr.OBSERVATION_USAGE_DETAILS]) == {
            "input": 5,
            "output": 7,
        }

    @pytest.mark.parametrize(
        ("observation", "name", "expected"),
        [
            (ToolRun(tool_name="t"), "execute_tool", "tool"),
            (AgentRun(agent_name="a"), "invoke_agent", "agent"),
            (
                GuardrailCheck(guardrail_name="g", stage="input"),
                "guardrail",
                "guardrail",
            ),
            (PipelineRun(name="p"), "pipeline", "chain"),
            (Other(label="x"), "something_else", "span"),
        ],
    )
    def test_each_observation_exports_with_its_langfuse_type(
        self, exported, observation, name, expected
    ):
        with observe(observation):
            pass

        assert kind(exported.get(name)) == expected


class TestEvents:
    def test_an_emitted_fact_lands_as_metadata_on_the_open_span(self, exported):
        with observe(PipelineRun(name="rag.query")):
            emit(Retrieval(top_k=3, hits=2))

        (span,) = exported.spans()
        assert span.name == "pipeline"
        assert metadata(span, "retrieval.top_k") == 3
        assert metadata(span, "retrieval.hits") == 2

    def test_a_cost_on_a_non_generation_stays_in_metadata(self, exported):
        with observe(AgentRun(agent_name="a")):
            emit(ModelCost(agent_name="a", model="gpt-4o", cost=0.5))

        span = exported.get("invoke_agent")
        assert metadata(span, "llm.request.model") == "gpt-4o"
        assert metadata(span, "llm.usage.cost") == 0.5
        assert Attr.OBSERVATION_COST_DETAILS not in span.attributes
        assert Attr.OBSERVATION_MODEL not in span.attributes

    def test_a_cost_on_a_generation_becomes_cost_details(self, exported):
        with observe(LLMCall(provider="openai", request_model="m")):
            emit(ModelCost(agent_name="a", model="m", cost=0.25))

        span = exported.get("chat")
        assert json.loads(span.attributes[Attr.OBSERVATION_COST_DETAILS]) == {
            "total": 0.25
        }


def everything_exported(span) -> str:
    """Every attribute, event and status description of a span, as one string."""
    events = [(e.name, dict(e.attributes or {})) for e in span.events]
    return f"{dict(span.attributes)} {events} {span.status.description}"


class TestErrors:
    def test_a_raised_error_exports_only_its_type_without_capture(self, exported):
        with pytest.raises(ValueError):
            with observe(ToolRun(tool_name="t")):
                raise ValueError("secret-text")

        span = exported.get("execute_tool")
        assert span.attributes[Attr.OBSERVATION_LEVEL] == "ERROR"
        assert span.attributes[Attr.OBSERVATION_STATUS_MESSAGE] == "ValueError"
        assert metadata(span, "error.type") == "ValueError"
        assert "secret-text" not in everything_exported(span)

    def test_a_raised_error_exports_its_message_with_capture(
        self, exported, monkeypatch
    ):
        monkeypatch.setenv("AGENT_PLATFORM_CAPTURE_CONTENT", "true")

        with pytest.raises(ValueError):
            with observe(ToolRun(tool_name="t")):
                raise ValueError("secret-text")

        span = exported.get("execute_tool")
        assert span.attributes[Attr.OBSERVATION_LEVEL] == "ERROR"
        assert (
            span.attributes[Attr.OBSERVATION_STATUS_MESSAGE]
            == "ValueError: secret-text"
        )
        assert metadata(span, "error.type") == "ValueError"

    def test_a_swallowed_error_exports_only_its_type_without_capture(self, exported):
        with observe(ToolRun(tool_name="t")) as run:
            run.record_error(RuntimeError("secret-text"))

        span = exported.get("execute_tool")
        assert span.attributes[Attr.OBSERVATION_LEVEL] == "ERROR"
        assert span.attributes[Attr.OBSERVATION_STATUS_MESSAGE] == "RuntimeError"
        assert metadata(span, "error.type") == "RuntimeError"
        assert "secret-text" not in everything_exported(span)

    def test_a_swallowed_error_exports_the_message_with_capture(
        self, exported, monkeypatch
    ):
        monkeypatch.setenv("AGENT_PLATFORM_CAPTURE_CONTENT", "true")

        with observe(ToolRun(tool_name="t")) as run:
            run.record_error(RuntimeError("denied"))

        span = exported.get("execute_tool")
        assert span.attributes[Attr.OBSERVATION_LEVEL] == "ERROR"
        assert span.attributes[Attr.OBSERVATION_STATUS_MESSAGE] == "denied"
        assert metadata(span, "error.type") == "RuntimeError"

    def test_the_caller_still_receives_the_original_exception(self, exported):
        original = ValueError("secret-text")

        with pytest.raises(ValueError) as caught:
            with observe(ToolRun(tool_name="t")):
                raise original

        assert caught.value is original

    def test_cancellation_still_propagates(self, exported):
        with pytest.raises(asyncio.CancelledError):
            with observe(ToolRun(tool_name="t")):
                raise asyncio.CancelledError()


class TestPayloadContent:
    def test_content_goes_to_input_and_output_when_capture_is_on(
        self, exported, monkeypatch
    ):
        monkeypatch.setenv("AGENT_PLATFORM_CAPTURE_CONTENT", "true")

        with observe(ToolRun(tool_name="t", arguments={"q": "Paris"})) as run:
            run.result = "sunny"

        span = exported.get("execute_tool")
        assert json.loads(span.attributes[Attr.OBSERVATION_INPUT]) == {"q": "Paris"}
        assert span.attributes[Attr.OBSERVATION_OUTPUT] == "sunny"

    def test_only_size_is_exported_without_capture(self, exported):
        with observe(ToolRun(tool_name="t", arguments={"q": "Paris"})) as run:
            run.result = "sunny"

        span = exported.get("execute_tool")
        assert Attr.OBSERVATION_INPUT not in span.attributes
        assert Attr.OBSERVATION_OUTPUT not in span.attributes
        assert metadata(span, "tool.arguments.size") > 0
        assert metadata(span, "tool.result.size") > 0
        digest_keys = [key for key in span.attributes if key.endswith(".digest")]
        assert digest_keys == []

    def test_digest_lands_in_metadata_when_capture_is_on(self, exported, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_CAPTURE_CONTENT", "true")

        with observe(ToolRun(tool_name="t", arguments={"q": "Paris"})) as run:
            run.result = "sunny"

        span = exported.get("execute_tool")
        assert metadata(span, "tool.arguments.digest")
        assert metadata(span, "tool.result.digest")


class TestNesting:
    def test_a_tool_inside_an_agent_has_the_agent_as_parent(self, exported):
        with observe(AgentRun(agent_name="a")):
            with observe(ToolRun(tool_name="t")):
                pass

        agent = exported.get("invoke_agent")
        tool = exported.get("execute_tool")
        assert tool.parent is not None
        assert tool.parent.span_id == agent.context.span_id

    def test_the_span_exposes_the_backend_ids(self, exported):
        with observe(PipelineRun(name="p")):
            span = current_span()
            trace_id, span_id = span.trace_id, span.span_id

        exported_span = exported.get("pipeline")
        assert trace_id == f"{exported_span.context.trace_id:032x}"
        assert span_id == f"{exported_span.context.span_id:016x}"


class TestCurrentSpan:
    def test_outside_any_span_it_is_inert(self, exported):
        span = current_span()

        assert isinstance(span, NoOpSpan)
        assert span.trace_id is None


class TestWithoutKeys:
    @pytest.mark.parametrize("missing", ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"])
    def test_a_missing_key_builds_no_client_and_records_nothing(
        self, monkeypatch, mocker, caplog, missing
    ):
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            if name != missing:
                monkeypatch.setenv(name, "set")
        constructor = mocker.patch(
            "agent_platform.integrations.tracing.langfuse.provider.Langfuse"
        )

        with caplog.at_level(logging.DEBUG):
            tracer = LangfuseTracer()
            set_tracer(tracer)
            try:
                with observe(PipelineRun(name="p")):
                    assert current_span().trace_id is None
            finally:
                reset_tracer()

        constructor.assert_not_called()
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_both_keys_build_the_client_from_the_environment(self, monkeypatch, mocker):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
        constructor = mocker.patch(
            "agent_platform.integrations.tracing.langfuse.provider.Langfuse"
        )

        LangfuseTracer()

        constructor.assert_called_once_with()
