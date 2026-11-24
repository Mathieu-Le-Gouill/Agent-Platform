import pytest
from pydantic import BaseModel

from agent_platform.agents.tools import current_tool_call
from agent_platform.agents.tools.base import Tool
from agent_platform.core.persistence import InMemoryCheckpointer
from agent_platform.core.schemas.message import ToolMessage
from agent_platform.workflows.graph import END, WorkflowGraph, current_step
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

    @pytest.mark.asyncio
    async def test_long_string_result_reaches_the_workflow_uncut(self):
        class _LongTool(Tool):
            name = "long"
            description = "Returns a very long string"
            input_schema = BaseModel

            async def run(self, **kwargs):
                return "x" * 50_000

        node = tool_node(_LongTool(), input_fn=lambda s: {}, output_fn=_output_fn)

        new_state = await node(WeatherState())

        assert new_state.forecast == "x" * 50_000


class _KeyRecordingTool(Tool):
    name = "record_key"
    description = "Records its idempotency key"
    input_schema = BaseModel

    def __init__(self) -> None:
        self.keys: list[str] = []

    async def run(self, **kwargs):
        self.keys.append(current_tool_call().idempotency_key)
        return "ok"


class TestToolNodeIdempotencyKey:
    @pytest.mark.asyncio
    async def test_a_resumed_node_sees_the_key_of_the_crashed_run(self):
        tool = _KeyRecordingTool()
        crash = [True]

        def output_fn(state: WeatherState, message: ToolMessage) -> WeatherState:
            if crash[0]:
                crash[0] = False
                raise RuntimeError("crash after the tool ran")
            return state

        async def prepare(state: WeatherState) -> WeatherState:
            return state

        graph: WorkflowGraph[WeatherState] = WorkflowGraph(WeatherState)
        graph.add_node("prepare", prepare)
        graph.add_node(
            "act", tool_node(tool, input_fn=lambda s: {}, output_fn=output_fn)
        )
        graph.set_entry_point("prepare")
        graph.add_edge("prepare", "act")
        graph.add_edge("act", END)
        workflow = graph.compile()
        checkpointer = InMemoryCheckpointer()

        with pytest.raises(RuntimeError, match="crash"):
            async for _ in workflow.arun(
                WeatherState(), checkpointer=checkpointer, run_id="run-1"
            ):
                pass
        async for _ in workflow.resume("run-1", checkpointer):
            pass

        assert tool.keys == ["run-1:act:2:1", "run-1:act:2:1"]

    @pytest.mark.asyncio
    async def test_without_a_run_id_the_keys_differ(self):
        tool = _KeyRecordingTool()
        graph: WorkflowGraph[WeatherState] = WorkflowGraph(WeatherState)
        graph.add_node(
            "act",
            tool_node(tool, input_fn=lambda s: {}, output_fn=lambda s, m: s),
        )
        graph.set_entry_point("act")
        graph.add_edge("act", END)
        workflow = graph.compile()

        for _ in range(2):
            async for _state in workflow.arun(WeatherState()):
                pass

        assert len(set(tool.keys)) == 2

    @pytest.mark.asyncio
    async def test_two_tool_nodes_in_one_node_get_different_keys(self):
        tool = _KeyRecordingTool()
        first = tool_node(tool, input_fn=lambda s: {}, output_fn=lambda s, m: s)
        second = tool_node(tool, input_fn=lambda s: {}, output_fn=lambda s, m: s)

        async def both(state: WeatherState) -> WeatherState:
            state = await first(state)
            return await second(state)

        graph: WorkflowGraph[WeatherState] = WorkflowGraph(WeatherState)
        graph.add_node("both", both)
        graph.set_entry_point("both")
        graph.add_edge("both", END)

        async for _ in graph.compile().arun(WeatherState(), run_id="run-1"):
            pass

        assert tool.keys == ["run-1:both:1:1", "run-1:both:1:2"]

    @pytest.mark.asyncio
    async def test_the_context_is_not_visible_between_nodes(self):
        graph: WorkflowGraph[WeatherState] = WorkflowGraph(WeatherState)

        async def noop(state: WeatherState) -> WeatherState:
            return state

        graph.add_node("n", noop)
        graph.set_entry_point("n")
        graph.add_edge("n", END)

        async for _ in graph.compile().arun(WeatherState(), run_id="run-1"):
            assert current_step() is None
