import pytest

from agent_platform.agents.agent import Agent
from agent_platform.agents.errors import AgentMaxIterations, AgentThinkError
from agent_platform.agents.executor import AgentExecutor
from agent_platform.core.schemas.message import (
    UserMessage,
)
from tests.helpers import make_fake_llm_response


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
