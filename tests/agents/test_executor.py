import pytest

from agent_platform.agents.agent import Agent
from agent_platform.agents.errors import (
    AgentMaxIterations,
    AgentRecoveryExhausted,
    AgentThinkError,
)
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.interfaces.llm.structured import StructuredOutputError
from agent_platform.core.schemas.message import UserMessage
from agent_platform.core.schemas.token import TokenUsage
from tests.helpers import (
    make_fake_llm_response,
    make_fake_stream,
    make_text_stream_chunks,
)
from tests.helpers.agents import StrictWeatherTool, WeatherTool, invoke_agent_span


class _StructuredFailureGuardrail:
    """Output guardrail whose own structured call failed."""

    def __init__(self) -> None:
        self.error = StructuredOutputError(
            "nested",
            last_result=make_fake_llm_response(""),
            usage=TokenUsage.zero(),
        )

    async def before(self, ctx):
        return None

    async def after(self, ctx, result):
        raise self.error


def _weather_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WeatherTool())
    return registry


class TestExecutorNestedStructuredFailure:
    @pytest.mark.asyncio
    async def test_run_propagates_it(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Hello")
        guardrail = _StructuredFailureGuardrail()
        ex = AgentExecutor(Agent(name="test", llm=mock_llm, guardrails=[guardrail]))

        with pytest.raises(StructuredOutputError) as raised:
            await ex.run("Hi")

        assert raised.value is guardrail.error

    @pytest.mark.asyncio
    async def test_run_streaming_propagates_it(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["Hello"])
        )
        guardrail = _StructuredFailureGuardrail()
        ex = AgentExecutor(Agent(name="test", llm=mock_llm, guardrails=[guardrail]))

        with pytest.raises(StructuredOutputError) as raised:
            async for _ in ex.run_streaming("Hi"):
                pass

        assert raised.value is guardrail.error


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
        agent = Agent(
            name="weather-agent", llm=mock_llm, tool_registry=_weather_registry()
        )
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
        agent = Agent(name="loopy", llm=mock_llm, tool_registry=_weather_registry())
        ex = AgentExecutor(agent, max_iterations=3)

        with pytest.raises(
            AgentMaxIterations, match="exceeded max iterations"
        ) as caught:
            await ex.run("Weather?")

        assert caught.value.code == "agent_max_iterations"
        assert mock_llm.agenerate.await_count == 3

    @pytest.mark.asyncio
    async def test_think_error_propagates(self, mock_llm):
        mock_llm.agenerate.side_effect = RuntimeError("crash")
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)

        with pytest.raises(AgentThinkError) as caught:
            await ex.run("Hi")
        assert caught.value.code == "llm_error"

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
        agent = Agent(name="multi", llm=mock_llm, tool_registry=_weather_registry())
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


class TestExecutorLoopObservability:
    @pytest.mark.asyncio
    async def test_a_direct_answer_exits_after_one_iteration(
        self, mock_llm, recorded_spans
    ):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Hi")
        ex = AgentExecutor(Agent(name="test", llm=mock_llm), max_iterations=4)

        await ex.run("Hi")

        span = invoke_agent_span(recorded_spans)
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
        ex = AgentExecutor(
            Agent(name="test", llm=mock_llm, tool_registry=_weather_registry())
        )

        await ex.run("Weather?")

        span = invoke_agent_span(recorded_spans)
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
        ex = AgentExecutor(
            Agent(name="test", llm=mock_llm, tool_registry=_weather_registry()),
            max_iterations=2,
        )

        with pytest.raises(AgentMaxIterations):
            await ex.run("Weather?")

        span = invoke_agent_span(recorded_spans)
        assert span.attributes["agent.exit_reason"] == "max_iterations"
        assert span.attributes["agent.iterations"] == 2
        assert span.attributes["agent.name"] == "test"

    @pytest.mark.asyncio
    async def test_failed_recovery_is_recorded(self, mock_llm, recorded_spans):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="", tool_calls=[{"id": "c1", "name": "strict_weather", "args": {}}]
        )
        registry = ToolRegistry()
        registry.register(StrictWeatherTool())
        ex = AgentExecutor(Agent(name="test", llm=mock_llm, tool_registry=registry))

        with pytest.raises(AgentRecoveryExhausted):
            await ex.run("Weather?")

        span = invoke_agent_span(recorded_spans)
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

        span = invoke_agent_span(recorded_spans)
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

        span = invoke_agent_span(recorded_spans)
        assert span.attributes["agent.exit_reason"] == "stop"
        assert span.attributes["agent.iterations"] == 1
