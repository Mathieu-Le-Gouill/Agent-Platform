import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agent_platform.agents.agent import Agent
from agent_platform.agents.errors import AgentThinkError
from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.schemas.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from agent_platform.core.schemas.token import TokenUsage
from tests.helpers import make_fake_llm_response


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
