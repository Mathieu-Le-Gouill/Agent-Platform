import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agent_platform.agents.agent import Agent
from agent_platform.agents.errors import AgentGuardrailError, AgentThinkError
from agent_platform.agents.guardrails import OutputNotEmptyGuardrail
from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.registry import ToolRegistry, is_tool_validation_error
from agent_platform.core.cost import ModelPricing, StaticPricingTable
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    ResponseFormat,
    StreamChunk,
)
from agent_platform.core.schemas.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.token_usage import TokenUsageAggregator
from tests.helpers import (
    make_fake_llm_response,
    make_fake_stream,
    make_text_stream_chunks,
)


class _WeatherInput(BaseModel):
    location: str


class _WeatherTool(Tool):
    name = "get_weather"
    description = "Get weather for a location"
    input_schema = _WeatherInput

    async def run(self, **kwargs):
        return {"temp": 22, "conditions": "sunny"}


class _FailingTool(Tool):
    name = "failing_tool"
    description = "A tool that always fails"
    input_schema = BaseModel

    async def run(self, **kwargs):
        raise RuntimeError("internal failure")


class _EmptyResultTool(Tool):
    name = "empty_tool"
    description = "A tool that returns None"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return None


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(_WeatherTool())
    return r


@pytest.fixture
def registry_with_failing():
    r = ToolRegistry()
    r.register(_FailingTool())
    return r


@pytest.fixture
def registry_with_empty():
    r = ToolRegistry()
    r.register(_EmptyResultTool())
    return r


@pytest.fixture
def agent(mock_llm, registry):
    return Agent(
        name="test-agent",
        llm=mock_llm,
        tool_registry=registry,
        model="test-model",
    )


@pytest.fixture
def agent_with_system_prompt(mock_llm, registry):
    return Agent(
        name="system-agent",
        llm=mock_llm,
        tool_registry=registry,
        system_prompt="You are a helpful test agent.",
        model="test-model",
    )


class TestAgentConstruction:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            Agent(name="", llm=MagicMock())

    def test_default_tool_registry(self):
        a = Agent(name="default", llm=MagicMock())
        assert isinstance(a.tool_registry, ToolRegistry)
        assert len(a.tool_registry) == 0

    def test_name_property(self, agent):
        assert agent.name == "test-agent"

    def test_tool_registry_property(self, agent, registry):
        assert agent.tool_registry is registry

    def test_system_prompt_property(self, agent_with_system_prompt):
        assert agent_with_system_prompt.system_prompt == "You are a helpful test agent."

    def test_no_system_prompt(self, agent):
        assert agent.system_prompt is None

    def test_no_response_schema_by_default(self, agent):
        assert agent.response_schema is None

    def test_no_cost_estimator_by_default(self, agent):
        assert agent.estimated_cost is None


class TestAgentEstimatedCost:
    @pytest.mark.asyncio
    async def test_computes_cost_from_recorded_usage(self, mock_llm, registry):
        pricing = StaticPricingTable(
            {"test-model": ModelPricing(input_per_1k=1.0, output_per_1k=2.0)}
        )
        aggregator = TokenUsageAggregator()
        agent = Agent(
            name="priced-agent",
            llm=mock_llm,
            tool_registry=registry,
            model="test-model",
            token_usage_aggregator=aggregator,
            cost_estimator=pricing,
        )
        mock_llm.agenerate.return_value = LLMResponse(
            message=AssistantMessage(content="hi"),
            usage=TokenUsage(input_tokens=1000, output_tokens=500),
            model="test-model",
            finish_reason=FinishReason.STOP,
        )

        await agent.think([UserMessage(content="hi")])

        assert agent.estimated_cost == pytest.approx(1.0 + 1.0)

    def test_unpriced_model_returns_zero(self, mock_llm, registry):
        pricing = StaticPricingTable({})
        agent = Agent(
            name="priced-agent",
            llm=mock_llm,
            tool_registry=registry,
            model="unknown-model",
            token_usage_aggregator=TokenUsageAggregator(),
            cost_estimator=pricing,
        )

        assert agent.estimated_cost == 0.0


class TestAgentThink:
    @pytest.mark.asyncio
    async def test_think_returns_assistant_message(self, agent, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Hello!")
        result = await agent.think([UserMessage(content="Hi")])
        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello!"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_think_with_tool_calls(self, agent, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "get_weather", "args": {"location": "Paris"}},
            ],
        )
        result = await agent.think([UserMessage(content="Weather?")])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"location": "Paris"}

    @pytest.mark.asyncio
    async def test_think_passes_tool_objects(self, agent, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Done")
        await agent.think([UserMessage(content="Hi")])
        call_args = mock_llm.agenerate.call_args[1]
        assert "tools" in call_args
        tools = call_args["tools"]
        assert len(tools) == 1
        assert tools[0].name == "get_weather"
        assert hasattr(tools[0], "run")

    @pytest.mark.asyncio
    async def test_think_no_tools_no_schemas(self, mock_llm):
        empty_registry = ToolRegistry()
        a = Agent(name="no-tools", llm=mock_llm, tool_registry=empty_registry)
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Done")
        await a.think([UserMessage(content="Hi")])
        call_kwargs = mock_llm.agenerate.call_args[1]
        assert call_kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_think_includes_system_prompt(
        self, agent_with_system_prompt, mock_llm
    ):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Sure")
        await agent_with_system_prompt.think([UserMessage(content="Hi")])
        call_args = mock_llm.agenerate.call_args[1]
        prompt = call_args["prompt"]
        assert prompt.system_prompt() == "You are a helpful test agent."

    @pytest.mark.asyncio
    async def test_think_llm_error_wrapped(self, agent, mock_llm):
        mock_llm.agenerate.side_effect = RuntimeError("LLM crashed")
        with pytest.raises(AgentThinkError, match="LLM generation failed"):
            await agent.think([UserMessage(content="Hi")])

    @pytest.mark.asyncio
    async def test_think_none_message_raises(self, agent, mock_llm):
        mock_llm.agenerate.return_value = LLMResponse(
            message=None,
            usage=TokenUsage(),
            model="test",
        )
        with pytest.raises(AgentThinkError, match="empty response"):
            await agent.think([UserMessage(content="Hi")])

    @pytest.mark.asyncio
    async def test_think_cancelled_error_propagates(self, agent, mock_llm):
        mock_llm.agenerate.side_effect = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await agent.think([UserMessage(content="Hi")])


class TestAgentAct:
    @pytest.mark.asyncio
    async def test_act_no_tool_calls_returns_empty(self, agent):
        msg = AssistantMessage(content="No tools needed")
        result = await agent.act(msg)
        assert result == []

    @pytest.mark.asyncio
    async def test_act_single_tool_call(self, agent, mock_llm):
        msg = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1", name="get_weather", arguments={"location": "Paris"}
                ),
            ],
        )
        results = await agent.act(msg)
        assert len(results) == 1
        tm = results[0]
        assert isinstance(tm, ToolMessage)
        assert tm.result.name == "get_weather"
        assert tm.result.is_error is False
        assert "sunny" in tm.result.content

    @pytest.mark.asyncio
    async def test_act_multiple_tool_calls(self, mock_llm):
        r = ToolRegistry()
        r.register(_WeatherTool())
        agent = Agent(name="multi", llm=mock_llm, tool_registry=r)
        msg = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="get_weather", arguments={"location": "Paris"}),
                ToolCall(id="c2", name="get_weather", arguments={"location": "London"}),
            ],
        )
        results = await agent.act(msg)
        assert len(results) == 2
        assert results[0].result.tool_call_id == "c1"
        assert results[1].result.tool_call_id == "c2"

    @pytest.mark.asyncio
    async def test_act_tool_error_captured(self, registry_with_failing, mock_llm):
        agent = Agent(name="failing", llm=mock_llm, tool_registry=registry_with_failing)
        msg = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="failing_tool", arguments={}),
            ],
        )
        results = await agent.act(msg)
        assert len(results) == 1
        assert results[0].result.is_error is True
        assert "internal failure" in results[0].result.content

    @pytest.mark.asyncio
    async def test_act_runs_tool_calls_concurrently(self, mock_llm):
        started = asyncio.Event()
        release = asyncio.Event()

        class _BlockingTool(Tool):
            name = "blocking"
            description = "Waits for a sibling call before completing"
            input_schema = BaseModel

            async def run(self, **kwargs):
                started.set()
                await asyncio.wait_for(release.wait(), timeout=1)
                return "done"

        class _UnblockingTool(Tool):
            name = "unblocking"
            description = "Waits for the sibling to start, then releases it"
            input_schema = BaseModel

            async def run(self, **kwargs):
                await asyncio.wait_for(started.wait(), timeout=1)
                release.set()
                return "released"

        r = ToolRegistry()
        r.register(_BlockingTool())
        r.register(_UnblockingTool())
        agent = Agent(name="concurrent", llm=mock_llm, tool_registry=r)
        msg = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="blocking", arguments={}),
                ToolCall(id="c2", name="unblocking", arguments={}),
            ],
        )
        results = await asyncio.wait_for(agent.act(msg), timeout=1)
        assert results[0].result.content == "done"
        assert results[1].result.content == "released"

    @pytest.mark.asyncio
    async def test_act_tool_returns_none(self, registry_with_empty, mock_llm):
        agent = Agent(name="empty", llm=mock_llm, tool_registry=registry_with_empty)
        msg = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="empty_tool", arguments={}),
            ],
        )
        results = await agent.act(msg)
        assert len(results) == 1
        assert results[0].result.content == ""


class _StreamingWeatherTool(Tool):
    name = "stream_weather"
    description = "Streams weather updates"
    input_schema = _WeatherInput
    supports_streaming = True

    async def astream(self, **kwargs):
        yield "22 degrees, "
        yield "sunny"


class _FailingStreamingTool(Tool):
    name = "failing_stream"
    description = "Streaming tool that fails"
    input_schema = BaseModel
    supports_streaming = True

    async def astream(self, **kwargs):
        yield "partial"
        raise RuntimeError("stream broke")


class TestAgentActStream:
    @pytest.mark.asyncio
    async def test_validation_error_tagged_on_reconstructed_message(self, mock_llm):
        class _StrictInput(BaseModel):
            location: str

        class _StrictTool(Tool):
            name = "strict_weather"
            description = "Requires a location"
            input_schema = _StrictInput

            async def run(self, **kwargs):
                return kwargs["location"]

        r = ToolRegistry()
        r.register(_StrictTool())
        agent = Agent(name="strict-agent", llm=mock_llm, tool_registry=r)
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="call_1", name="strict_weather", arguments={})],
        )
        events = [e async for e in agent.act_stream(msg)]
        tool_messages = [e for e in events if isinstance(e, ToolMessage)]
        assert len(tool_messages) == 1
        assert is_tool_validation_error(tool_messages[0]) is True

    @pytest.mark.asyncio
    async def test_non_streaming_tool_call(self, agent):
        msg = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1", name="get_weather", arguments={"location": "Paris"}
                ),
            ],
        )
        events = [e async for e in agent.act_stream(msg)]
        tool_messages = [e for e in events if isinstance(e, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].result.is_error is False
        assert "sunny" in tool_messages[0].result.content

    @pytest.mark.asyncio
    async def test_streaming_tool_accumulates_content(self, mock_llm):
        r = ToolRegistry()
        r.register(_StreamingWeatherTool())
        agent = Agent(name="streamer", llm=mock_llm, tool_registry=r)
        msg = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="stream_weather",
                    arguments={"location": "Paris"},
                ),
            ],
        )
        events = [e async for e in agent.act_stream(msg)]
        tool_messages = [e for e in events if isinstance(e, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].result.content == "22 degrees, sunny"
        assert tool_messages[0].result.is_error is False

    @pytest.mark.asyncio
    async def test_streaming_tool_error_captured(self, mock_llm):
        r = ToolRegistry()
        r.register(_FailingStreamingTool())
        agent = Agent(name="failing-streamer", llm=mock_llm, tool_registry=r)
        msg = AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="call_1", name="failing_stream", arguments={}),
            ],
        )
        events = [e async for e in agent.act_stream(msg)]
        tool_messages = [e for e in events if isinstance(e, ToolMessage)]
        assert len(tool_messages) == 1
        assert tool_messages[0].result.is_error is True
        assert "stream broke" in tool_messages[0].result.content


class TestAgentStep:
    @pytest.mark.asyncio
    async def test_step_no_tools(self, agent, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="Direct answer"
        )
        assistant_msg, tool_msgs = await agent.step([UserMessage(content="Hi")])
        assert assistant_msg.content == "Direct answer"
        assert tool_msgs == []

    @pytest.mark.asyncio
    async def test_step_with_tools(self, agent, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="",
            tool_calls=[
                {"id": "c1", "name": "get_weather", "args": {"location": "Paris"}},
            ],
        )
        assistant_msg, tool_msgs = await agent.step([UserMessage(content="Weather?")])
        assert len(tool_msgs) == 1
        assert tool_msgs[0].result.name == "get_weather"
        assert tool_msgs[0].result.is_error is False

    @pytest.mark.asyncio
    async def test_step_does_not_mutate_input(self, agent, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="Thinking",
            tool_calls=[
                {"id": "c1", "name": "get_weather", "args": {"location": "Paris"}},
            ],
        )
        msgs = [UserMessage(content="Weather?")]
        assistant_msg, tool_msgs = await agent.step(msgs)
        assert len(msgs) == 1
        assert msgs[0].content == "Weather?"


class TestAgentGuardrails:
    @pytest.mark.asyncio
    async def test_no_guardrails_by_default(self, mock_llm):
        agent = Agent(name="plain", llm=mock_llm)
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Hi")
        result = await agent.think([UserMessage(content="Hi")])
        assert result.content == "Hi"

    @pytest.mark.asyncio
    async def test_guardrail_rejects_empty_response(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="")
        agent = Agent(
            name="guarded", llm=mock_llm, guardrails=[OutputNotEmptyGuardrail()]
        )
        with pytest.raises(AgentGuardrailError, match="empty"):
            await agent.think([UserMessage(content="Hi")])

    @pytest.mark.asyncio
    async def test_guardrail_allows_non_empty_response(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="Fine")
        agent = Agent(
            name="guarded", llm=mock_llm, guardrails=[OutputNotEmptyGuardrail()]
        )
        result = await agent.think([UserMessage(content="Hi")])
        assert result.content == "Fine"

    @pytest.mark.asyncio
    async def test_before_short_circuit_skips_generation(self, mock_llm):
        class _ShortCircuit:
            async def before(self, ctx):
                return AssistantMessage(content="canned")

            async def after(self, ctx, result):
                return result

        agent = Agent(name="guarded", llm=mock_llm, guardrails=[_ShortCircuit()])
        result = await agent.think([UserMessage(content="Hi")])
        assert result.content == "canned"
        mock_llm.agenerate.assert_not_awaited()


class _Answer(BaseModel):
    value: int


class TestAgentResponseSchema:
    @pytest.mark.asyncio
    async def test_no_schema_returns_text_unvalidated(self, mock_llm):
        agent = Agent(name="plain", llm=mock_llm)
        mock_llm.agenerate.return_value = make_fake_llm_response(content="not json")
        result = await agent.think([UserMessage(content="Hi")])
        assert result.text == "not json"

    @pytest.mark.asyncio
    async def test_valid_json_passes_on_first_try(self, mock_llm):
        agent = Agent(name="structured", llm=mock_llm, response_schema=_Answer)
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content='{"value": 42}'
        )
        result = await agent.think([UserMessage(content="Hi")])
        assert result.text == '{"value": 42}'
        assert mock_llm.agenerate.await_count == 1

    @pytest.mark.asyncio
    async def test_invalid_json_recovers_on_retry(self, mock_llm):
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_llm_response(content="not json")
            return make_fake_llm_response(content='{"value": 7}')

        mock_llm.agenerate.side_effect = gen
        agent = Agent(name="structured", llm=mock_llm, response_schema=_Answer)
        result = await agent.think([UserMessage(content="Hi")])
        assert result.text == '{"value": 7}'
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_invalid_json_twice_gives_up_with_last_result(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="still not json"
        )
        agent = Agent(name="structured", llm=mock_llm, response_schema=_Answer)
        result = await agent.think([UserMessage(content="Hi")])
        assert result.text == "still not json"
        assert mock_llm.agenerate.await_count == 2

    @pytest.mark.asyncio
    async def test_tool_calls_skip_schema_validation(self, mock_llm, registry):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="",
            tool_calls=[
                {"id": "c1", "name": "get_weather", "args": {"location": "Paris"}}
            ],
        )
        agent = Agent(
            name="structured",
            llm=mock_llm,
            tool_registry=registry,
            response_schema=_Answer,
        )
        result = await agent.think([UserMessage(content="Weather?")])
        assert len(result.tool_calls) == 1
        assert mock_llm.agenerate.await_count == 1

    @pytest.mark.asyncio
    async def test_does_not_mutate_input_messages(self, mock_llm):
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_llm_response(content="not json")
            return make_fake_llm_response(content='{"value": 1}')

        mock_llm.agenerate.side_effect = gen
        agent = Agent(name="structured", llm=mock_llm, response_schema=_Answer)
        msgs = [UserMessage(content="Hi")]
        await agent.think(msgs)
        assert len(msgs) == 1

    @pytest.mark.asyncio
    async def test_sets_json_schema_response_format_on_generation_config(
        self, mock_llm
    ):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content='{"value": 42}'
        )
        agent = Agent(name="structured", llm=mock_llm, response_schema=_Answer)
        await agent.think([UserMessage(content="Hi")])
        config = mock_llm.agenerate.call_args[1]["config"]
        assert config.response_format is ResponseFormat.JSON_SCHEMA
        assert config.json_schema == _Answer.model_json_schema()

    @pytest.mark.asyncio
    async def test_does_not_override_explicit_response_format(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="not json")
        agent = Agent(
            name="structured",
            llm=mock_llm,
            response_schema=_Answer,
            generation_config=GenerationConfig(response_format=ResponseFormat.JSON),
        )
        await agent.think([UserMessage(content="Hi")])
        config = mock_llm.agenerate.call_args[1]["config"]
        assert config.response_format is ResponseFormat.JSON
        assert config.json_schema is None


class TestAgentThinkStream:
    @pytest.mark.asyncio
    async def test_think_stream_yields_chunks(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["Hel", "lo"])
        )
        agent = Agent(name="streamer", llm=mock_llm)
        chunks = [c async for c in agent.think_stream([UserMessage(content="Hi")])]
        deltas = [c.delta for c in chunks if c.delta]
        assert deltas == ["Hel", "lo"]

    @pytest.mark.asyncio
    async def test_think_stream_wraps_errors(self, mock_llm):
        def boom(**kwargs):
            raise RuntimeError("stream crashed")

        mock_llm.stream.side_effect = boom
        agent = Agent(name="streamer", llm=mock_llm)
        with pytest.raises(AgentThinkError, match="LLM streaming failed"):
            async for _ in agent.think_stream([UserMessage(content="Hi")]):
                pass

    @pytest.mark.asyncio
    async def test_think_stream_cancelled_error_propagates(self, mock_llm):
        async def cancelling_gen(**kwargs):
            raise asyncio.CancelledError()
            yield  # pragma: no cover

        mock_llm.stream.side_effect = cancelling_gen
        agent = Agent(name="streamer", llm=mock_llm)
        with pytest.raises(asyncio.CancelledError):
            async for _ in agent.think_stream([UserMessage(content="Hi")]):
                pass


class TestAgentTokenUsage:
    def test_zero_usage_without_aggregator(self, agent):
        assert agent.token_usage == TokenUsage.zero()

    @pytest.mark.asyncio
    async def test_think_records_usage_under_usage_key(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response("hi")
        aggregator = TokenUsageAggregator()
        agent = Agent(
            name="metered",
            llm=mock_llm,
            token_usage_aggregator=aggregator,
        )

        await agent.think([UserMessage(content="hi")])

        assert agent.token_usage == TokenUsage(input_tokens=10, output_tokens=5)
        assert aggregator.total_for("metered") == TokenUsage(
            input_tokens=10, output_tokens=5
        )

    @pytest.mark.asyncio
    async def test_think_accumulates_across_calls(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response("hi")
        agent = Agent(
            name="metered",
            llm=mock_llm,
            token_usage_aggregator=TokenUsageAggregator(),
        )

        await agent.think([UserMessage(content="hi")])
        await agent.think([UserMessage(content="hi again")])

        assert agent.token_usage == TokenUsage(input_tokens=20, output_tokens=10)

    @pytest.mark.asyncio
    async def test_think_uses_explicit_usage_key(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response("hi")
        aggregator = TokenUsageAggregator()
        agent = Agent(
            name="metered",
            llm=mock_llm,
            token_usage_aggregator=aggregator,
            usage_key="conversation-42",
        )

        await agent.think([UserMessage(content="hi")])

        assert aggregator.total_for("conversation-42") == TokenUsage(
            input_tokens=10, output_tokens=5
        )
        assert aggregator.total_for("metered") == TokenUsage.zero()

    @pytest.mark.asyncio
    async def test_think_stream_records_usage_from_final_chunk(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            [
                StreamChunk(delta="Hi"),
                StreamChunk(
                    delta="",
                    finish_reason=FinishReason.STOP,
                    usage=TokenUsage(input_tokens=4, output_tokens=1),
                ),
            ]
        )
        aggregator = TokenUsageAggregator()
        agent = Agent(name="streamer", llm=mock_llm, token_usage_aggregator=aggregator)

        async for _ in agent.think_stream([UserMessage(content="Hi")]):
            pass

        assert aggregator.total_for("streamer") == TokenUsage(
            input_tokens=4, output_tokens=1
        )
