import pytest
from pydantic import BaseModel

from agent_platform.agents.agent import Agent
from agent_platform.agents.errors import (
    AgentGuardrailError,
    AgentMaxIterations,
    AgentThinkError,
)
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools.base import Tool, ToolStreamChunk
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.schemas.message import (
    AssistantMessage,
    UserMessage,
)
from tests.helpers import (
    make_fake_llm_response,
    make_fake_stream,
    make_text_stream_chunks,
    make_tool_call_stream_chunks,
)


class _WeatherTool(Tool):
    name = "get_weather"
    description = "Get weather"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return {"temp": 22}


class _Answer(BaseModel):
    value: int


class _BlockingGuardrail:
    async def before(self, ctx):
        return None

    async def after(self, ctx, result):
        raise AgentGuardrailError("output blocked")


class _RewritingGuardrail:
    async def before(self, ctx):
        return None

    async def after(self, ctx, result):
        return AssistantMessage(content="[redacted]")


@pytest.fixture
def agent(mock_llm):
    return Agent(name="test", llm=mock_llm)


class TestExecutorConstruction:
    def test_default_max_iterations(self, mock_llm):
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)
        assert ex.max_iterations == 10

    def test_custom_max_iterations(self, mock_llm):
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent, max_iterations=5)
        assert ex.max_iterations == 5

    def test_invalid_max_iterations_raises(self, mock_llm):
        agent = Agent(name="test", llm=mock_llm)
        with pytest.raises(ValueError, match="max_iterations"):
            AgentExecutor(agent, max_iterations=0)

    def test_agent_property(self, mock_llm):
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)
        assert ex.agent is agent


class TestExecutorRun:
    @pytest.mark.asyncio
    async def test_simple_response(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Hello world")
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)
        result = await ex.run("Hi")
        assert result == "Hello world"
        assert mock_llm.agenerate.await_count == 1

    @pytest.mark.asyncio
    async def test_single_tool_round(self, mock_llm):
        call_count = 0

        async def generate_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_llm_response(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "get_weather",
                            "args": {"location": "Paris"},
                        },
                    ],
                )
            return make_fake_llm_response(content="It's 22°C and sunny in Paris!")

        mock_llm.agenerate.side_effect = generate_side_effect

        from pydantic import BaseModel

        from agent_platform.agents.tools.base import Tool
        from agent_platform.agents.tools.registry import ToolRegistry

        class _WeatherTool(Tool):
            name = "get_weather"
            description = "Get weather"
            input_schema = BaseModel

            async def run(self, **kwargs):
                return {"temp": 22, "conditions": "sunny"}

        registry = ToolRegistry()
        registry.register(_WeatherTool())

        agent = Agent(name="weather-agent", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)
        result = await ex.run("What's the weather in Paris?")
        assert "22°C" in result
        assert mock_llm.agenerate.await_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="",
            tool_calls=[
                {"id": "c1", "name": "get_weather", "args": {"location": "Paris"}},
            ],
        )

        from pydantic import BaseModel

        from agent_platform.agents.tools.base import Tool
        from agent_platform.agents.tools.registry import ToolRegistry

        class _WeatherTool(Tool):
            name = "get_weather"
            description = "Get weather"
            input_schema = BaseModel

            async def run(self, **kwargs):
                return {"temp": 22}

        registry = ToolRegistry()
        registry.register(_WeatherTool())

        agent = Agent(name="loopy", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent, max_iterations=3)

        with pytest.raises(AgentMaxIterations, match="exceeded max iterations"):
            await ex.run("Weather?")

        assert mock_llm.agenerate.await_count == 3

    @pytest.mark.asyncio
    async def test_think_error_propagates(self, mock_llm):
        mock_llm.agenerate.side_effect = RuntimeError("crash")
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)

        with pytest.raises(AgentThinkError, match="LLM generation failed"):
            await ex.run("Hi")

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_round(self, mock_llm):
        call_count = 0

        async def generate_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_llm_response(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "get_weather",
                            "args": {"location": "Paris"},
                        },
                        {
                            "id": "c2",
                            "name": "get_weather",
                            "args": {"location": "London"},
                        },
                    ],
                )
            return make_fake_llm_response(content="Here are both forecasts.")

        mock_llm.agenerate.side_effect = generate_side_effect

        from pydantic import BaseModel

        from agent_platform.agents.tools.base import Tool
        from agent_platform.agents.tools.registry import ToolRegistry

        class _WeatherTool(Tool):
            name = "get_weather"
            description = "Get weather"
            input_schema = BaseModel

            async def run(self, **kwargs):
                return {"temp": 20}

        registry = ToolRegistry()
        registry.register(_WeatherTool())

        agent = Agent(name="multi", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)
        result = await ex.run("Weather in two cities?")
        assert "Here are both forecasts" in result
        assert mock_llm.agenerate.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_content_result(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="")
        agent = Agent(name="empty", llm=mock_llm)
        ex = AgentExecutor(agent)
        result = await ex.run("Say nothing")
        assert result == ""


class TestExecutorRunStreaming:
    @pytest.mark.asyncio
    async def test_direct_response_yields_text(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["Hello ", "world"])
        )
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)
        events = [e async for e in ex.run_streaming("Hi")]
        assert events == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_tool_round_then_final_answer(self, mock_llm):
        call_count = 0

        def stream_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_stream(
                    make_tool_call_stream_chunks(
                        [
                            {
                                "id": "c1",
                                "name": "get_weather",
                                "args": {"location": "Paris"},
                            }
                        ]
                    )
                )
            return make_fake_stream(make_text_stream_chunks(["It's sunny in Paris!"]))

        mock_llm.stream.side_effect = stream_side_effect

        registry = ToolRegistry()
        registry.register(_WeatherTool())
        agent = Agent(name="weather-agent", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        events = [e async for e in ex.run_streaming("Weather in Paris?")]
        assert events[-1] == "It's sunny in Paris!"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_tool_call_stream_chunks(
                [{"id": "c1", "name": "get_weather", "args": {"location": "Paris"}}]
            )
        )
        registry = ToolRegistry()
        registry.register(_WeatherTool())
        agent = Agent(name="loopy", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent, max_iterations=2)

        with pytest.raises(AgentMaxIterations, match="exceeded max iterations"):
            async for _ in ex.run_streaming("Weather?"):
                pass

    @pytest.mark.asyncio
    async def test_repeated_bad_calls_give_up_without_raising(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_tool_call_stream_chunks(
                [{"id": "c1", "name": "strict_weather", "args": {}}]
            )
        )
        registry = ToolRegistry()
        registry.register(_StrictWeatherTool())
        agent = Agent(name="stuck-streamer", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        events = [e async for e in ex.run_streaming("Weather?")]

        tool_events = [e for e in events if isinstance(e, ToolStreamChunk)]
        assert len(tool_events) == 1
        assert all(e.is_validation_error for e in tool_events)
        assert not any(isinstance(e, str) for e in events)

    @pytest.mark.asyncio
    async def test_blocking_output_guardrail_raises_without_yielding_text(
        self, mock_llm
    ):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["secret ", "text"])
        )
        agent = Agent(name="guarded", llm=mock_llm, guardrails=[_BlockingGuardrail()])
        ex = AgentExecutor(agent)
        events: list = []

        with pytest.raises(AgentGuardrailError):
            async for event in ex.run_streaming("Hi"):
                events.append(event)

        assert events == []

    @pytest.mark.asyncio
    async def test_rewriting_output_guardrail_yields_the_rewritten_text(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["secret ", "text"])
        )
        agent = Agent(name="guarded", llm=mock_llm, guardrails=[_RewritingGuardrail()])
        ex = AgentExecutor(agent)

        events = [e async for e in ex.run_streaming("Hi")]

        assert events == ["[redacted]"]

    @pytest.mark.asyncio
    async def test_invalid_response_schema_retry_yields_only_the_accepted_attempt(
        self, mock_llm
    ):
        attempts = [
            make_text_stream_chunks(["not ", "json"]),
            make_text_stream_chunks(['{"value":', " 1}"]),
        ]
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(attempts.pop(0))
        agent = Agent(name="structured", llm=mock_llm, response_schema=_Answer)
        ex = AgentExecutor(agent)

        events = [e async for e in ex.run_streaming("Hi")]

        assert events == ['{"value":', " 1}"]

    @pytest.mark.asyncio
    async def test_tool_call_retry_drops_the_first_attempts_events(self, mock_llm):
        attempts = [
            [
                *make_text_stream_chunks(["draft "])[:-1],
                *make_tool_call_stream_chunks(
                    [{"id": "c1", "name": "strict_weather", "args": {}}]
                ),
            ],
            make_tool_call_stream_chunks(
                [
                    {
                        "id": "c2",
                        "name": "strict_weather",
                        "args": {"location": "Paris"},
                    }
                ]
            ),
            make_text_stream_chunks(["Sunny"]),
        ]
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(attempts.pop(0))
        registry = ToolRegistry()
        registry.register(_StrictWeatherTool())
        agent = Agent(name="recovering", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        events = [e async for e in ex.run_streaming("Weather?")]

        assert "draft " not in events
        assert not any(
            isinstance(e, ToolStreamChunk) and e.is_validation_error for e in events
        )
        assert events[-1] == "Sunny"


class TestExecutorRunWithMessages:
    @pytest.mark.asyncio
    async def test_returns_accumulated_messages(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="Prebuilt answer"
        )
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)
        messages = [UserMessage(content="Prebuilt")]
        result, accumulated = await ex.run_with_messages(messages)
        assert result == "Prebuilt answer"
        assert len(accumulated) == 2
        assert accumulated[0] is messages[0]


class TestExecutorTracing:
    @pytest.mark.asyncio
    async def test_run_sets_conversation_id_when_given(self, mock_llm, recorded_spans):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Hi")
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)

        await ex.run("Hi", conversation_id="conv-123")

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["conversation.id"] == "conv-123"

    @pytest.mark.asyncio
    async def test_run_omits_conversation_id_when_not_given(
        self, mock_llm, recorded_spans
    ):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Hi")
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)

        await ex.run("Hi")

        (span,) = recorded_spans.get_finished_spans()
        assert "conversation.id" not in span.attributes

    @pytest.mark.asyncio
    async def test_run_streaming_sets_conversation_id_when_given(
        self, mock_llm, recorded_spans
    ):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["Hi"])
        )
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)

        async for _ in ex.run_streaming("Hi", conversation_id="conv-456"):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["conversation.id"] == "conv-456"


class _StrictWeatherInput(BaseModel):
    location: str


class _StrictWeatherTool(Tool):
    name = "strict_weather"
    description = "Get weather, requires a location"
    input_schema = _StrictWeatherInput

    async def run(self, **kwargs):
        return {"temp": 22, "location": kwargs["location"]}


class TestExecutorToolCallRecovery:
    @pytest.mark.asyncio
    async def test_single_bad_call_recovered_on_retry(self, mock_llm):
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_llm_response(
                    content="",
                    tool_calls=[{"id": "c1", "name": "strict_weather", "args": {}}],
                )
            if call_count == 2:
                return make_fake_llm_response(
                    content="",
                    tool_calls=[
                        {
                            "id": "c2",
                            "name": "strict_weather",
                            "args": {"location": "Paris"},
                        }
                    ],
                )
            return make_fake_llm_response(content="Sunny in Paris")

        mock_llm.agenerate.side_effect = gen
        registry = ToolRegistry()
        registry.register(_StrictWeatherTool())
        agent = Agent(name="recovering", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        result = await ex.run("Weather in Paris?")

        assert result == "Sunny in Paris"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_repeated_bad_calls_still_fail_after_one_retry(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="", tool_calls=[{"id": "c1", "name": "strict_weather", "args": {}}]
        )
        registry = ToolRegistry()
        registry.register(_StrictWeatherTool())
        agent = Agent(name="stuck", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        result, messages = await ex.run_with_messages(
            [UserMessage(content="Weather in Paris?")]
        )

        assert mock_llm.agenerate.await_count == 2
        assert any(
            isinstance(m, UserMessage) and "invalid" in m.content for m in messages
        )

    @pytest.mark.asyncio
    async def test_well_formed_calls_unaffected(self, mock_llm):
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_llm_response(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "strict_weather",
                            "args": {"location": "Paris"},
                        }
                    ],
                )
            return make_fake_llm_response(content="Sunny in Paris")

        mock_llm.agenerate.side_effect = gen
        registry = ToolRegistry()
        registry.register(_StrictWeatherTool())
        agent = Agent(name="fine", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        result = await ex.run("Weather in Paris?")

        assert result == "Sunny in Paris"
        assert call_count == 2


class TestAssembleToolCalls:
    def test_malformed_arguments_json_falls_back_to_empty_dict(self):
        from agent_platform.agents.agent import _assemble_tool_calls
        from agent_platform.core.interfaces.llm.response import ToolCallDelta

        deltas = [
            ToolCallDelta(
                index=0, id="c1", name="get_weather", arguments_delta="{not json"
            )
        ]

        calls = _assemble_tool_calls(deltas)

        assert len(calls) == 1
        assert calls[0].id == "c1"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {}


def _invoke_agent_span(recorded_spans):
    spans = [s for s in recorded_spans.get_finished_spans() if s.name == "invoke_agent"]
    assert len(spans) == 1
    return spans[0]


class TestExecutorLoopObservability:
    @pytest.mark.asyncio
    async def test_a_direct_answer_exits_after_one_iteration(
        self, mock_llm, recorded_spans
    ):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Hi")
        ex = AgentExecutor(Agent(name="test", llm=mock_llm), max_iterations=4)

        await ex.run("Hi")

        span = _invoke_agent_span(recorded_spans)
        assert span.attributes["agent.iterations"] == 1
        assert span.attributes["agent.max_iterations"] == 4
        assert span.attributes["agent.exit_reason"] == "stop"
        assert span.attributes["agent.recovery_attempted"] is False
        assert span.attributes["agent.tool_calls_requested"] == 0
        assert span.attributes["agent.tool_calls_executed"] == 0

    @pytest.mark.asyncio
    async def test_counts_the_tool_calls_a_turn_asked_for_and_ran(
        self, mock_llm, recorded_spans
    ):
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_llm_response(
                    content="",
                    tool_calls=[{"id": "c1", "name": "get_weather", "args": {}}],
                )
            return make_fake_llm_response(content="Sunny")

        mock_llm.agenerate.side_effect = gen
        registry = ToolRegistry()
        registry.register(_WeatherTool())
        ex = AgentExecutor(Agent(name="test", llm=mock_llm, tool_registry=registry))

        await ex.run("Weather?")

        span = _invoke_agent_span(recorded_spans)
        assert span.attributes["agent.iterations"] == 2
        assert span.attributes["agent.tool_calls_requested"] == 1
        assert span.attributes["agent.tool_calls_executed"] == 1

    @pytest.mark.asyncio
    async def test_running_out_of_iterations_is_recorded(
        self, mock_llm, recorded_spans
    ):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="", tool_calls=[{"id": "c1", "name": "get_weather", "args": {}}]
        )
        registry = ToolRegistry()
        registry.register(_WeatherTool())
        ex = AgentExecutor(
            Agent(name="test", llm=mock_llm, tool_registry=registry), max_iterations=2
        )

        with pytest.raises(AgentMaxIterations):
            await ex.run("Weather?")

        span = _invoke_agent_span(recorded_spans)
        assert span.attributes["agent.exit_reason"] == "max_iterations"
        assert span.attributes["agent.iterations"] == 2
        assert span.attributes["agent.name"] == "test"

    @pytest.mark.asyncio
    async def test_failed_recovery_is_recorded(self, mock_llm, recorded_spans):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="", tool_calls=[{"id": "c1", "name": "strict_weather", "args": {}}]
        )
        registry = ToolRegistry()
        registry.register(_StrictWeatherTool())
        ex = AgentExecutor(Agent(name="test", llm=mock_llm, tool_registry=registry))

        await ex.run("Weather?")

        span = _invoke_agent_span(recorded_spans)
        assert span.attributes["agent.exit_reason"] == "recovery_exhausted"
        assert span.attributes["agent.recovery_attempted"] is True
        assert span.attributes["agent.tool_calls_requested"] == 1

    @pytest.mark.asyncio
    async def test_an_unexpected_failure_exits_under_its_error_type(
        self, mock_llm, recorded_spans
    ):
        mock_llm.agenerate.side_effect = RuntimeError("boom")
        ex = AgentExecutor(Agent(name="test", llm=mock_llm))

        with pytest.raises(AgentThinkError):
            await ex.run("Hi")

        span = _invoke_agent_span(recorded_spans)
        assert span.attributes["agent.exit_reason"] == "AgentThinkError"

    @pytest.mark.asyncio
    async def test_streaming_records_the_same_loop_facts(
        self, mock_llm, recorded_spans
    ):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["Hi"])
        )
        ex = AgentExecutor(Agent(name="test", llm=mock_llm))

        async for _event in ex.run_streaming("Hi"):
            pass

        span = _invoke_agent_span(recorded_spans)
        assert span.attributes["agent.exit_reason"] == "stop"
        assert span.attributes["agent.iterations"] == 1
