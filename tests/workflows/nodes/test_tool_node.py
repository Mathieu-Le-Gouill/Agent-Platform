import pytest
from pydantic import BaseModel

from agent_platform.agents.tools.base import Tool
from agent_platform.core.schemas.message import ToolMessage
from agent_platform.workflows.nodes.tool_node import tool_node
from agent_platform.workflows.state import WorkflowState


class _WeatherInput(BaseModel):
    location: str


class _WeatherTool(Tool):
    name = "get_weather"
    description = "Get the weather for a location"
    input_schema = _WeatherInput

    async def run(self, **kwargs):
        return f"sunny in {kwargs['location']}"


class _FailingTool(Tool):
    name = "boom"
    description = "Always fails"
    input_schema = BaseModel

    async def run(self, **kwargs):
        raise RuntimeError("kaboom")


class WeatherState(WorkflowState):
    location: str = "Paris"
    forecast: str = ""


def _input_fn(state: WeatherState) -> dict:
    return {"location": state.location}


def _output_fn(state: WeatherState, message: ToolMessage) -> WeatherState:
    return state.model_copy(update={"forecast": message.result.content})


class TestToolNode:
    @pytest.mark.asyncio
    async def test_invokes_tool_and_writes_result_back(self):
        node = tool_node(_WeatherTool(), input_fn=_input_fn, output_fn=_output_fn)

        new_state = await node(WeatherState(location="Lyon"))

        assert new_state.forecast == "sunny in Lyon"

    @pytest.mark.asyncio
    async def test_tool_error_is_wrapped_not_raised(self):
        def output_fn(state: WeatherState, message: ToolMessage) -> WeatherState:
            assert message.result.is_error
            return state.model_copy(update={"forecast": "error"})

        node = tool_node(_FailingTool(), input_fn=lambda s: {}, output_fn=output_fn)

        new_state = await node(WeatherState())

        assert new_state.forecast == "error"
