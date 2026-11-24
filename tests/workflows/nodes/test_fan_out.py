import asyncio

import pytest
from pydantic import BaseModel

from agent_platform.agents.tools import current_tool_call
from agent_platform.agents.tools.base import Tool
from agent_platform.core.schemas.message import AssistantMessage
from agent_platform.workflows.graph import END, NodeFn, WorkflowGraph
from agent_platform.workflows.nodes.fan_out import fan_out
from agent_platform.workflows.nodes.tool_node import tool_node
from agent_platform.workflows.state import MessagesState


class TestFanOut:
    @pytest.mark.asyncio
    async def test_runs_branches_and_merges_results(self):
        async def branch_a(state: MessagesState) -> MessagesState:
            return state.model_copy(
                update={"messages": [*state.messages, AssistantMessage(content="a")]}
            )

        async def branch_b(state: MessagesState) -> MessagesState:
            return state.model_copy(
                update={"messages": [*state.messages, AssistantMessage(content="b")]}
            )

        def merge(state: MessagesState, results: list[MessagesState]) -> MessagesState:
            merged = [msg for result in results for msg in result.messages]
            return state.model_copy(update={"messages": merged})

        node = fan_out([("a", branch_a), ("b", branch_b)], merge)
        result = await node(MessagesState())

        assert [m.text for m in result.messages] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_branches_run_concurrently(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking(state: MessagesState) -> MessagesState:
            started.set()
            await asyncio.wait_for(release.wait(), timeout=1)
            return state.model_copy(
                update={"messages": [*state.messages, AssistantMessage(content="done")]}
            )

        async def unblocking(state: MessagesState) -> MessagesState:
            await asyncio.wait_for(started.wait(), timeout=1)
            release.set()
            return state.model_copy(
                update={
                    "messages": [*state.messages, AssistantMessage(content="released")]
                }
            )

        def merge(state: MessagesState, results: list[MessagesState]) -> MessagesState:
            merged = [msg for result in results for msg in result.messages]
            return state.model_copy(update={"messages": merged})

        node = fan_out([("blocking", blocking), ("unblocking", unblocking)], merge)
        result = await asyncio.wait_for(node(MessagesState()), timeout=1)

        assert [m.text for m in result.messages] == ["done", "released"]

    @pytest.mark.asyncio
    async def test_branches_receive_independent_state_copy(self):
        async def mutator(state: MessagesState) -> MessagesState:
            return state.model_copy(
                update={"messages": [AssistantMessage(content="mutated")]}
            )

        async def observer(state: MessagesState) -> MessagesState:
            return state

        def merge(state: MessagesState, results: list[MessagesState]) -> MessagesState:
            return results[1]

        initial = MessagesState(messages=[AssistantMessage(content="original")])
        node = fan_out([("mutator", mutator), ("observer", observer)], merge)
        result = await node(initial)

        assert [m.text for m in result.messages] == ["original"]


class _KeyTool(Tool):
    name = "record_key"
    description = "Records its idempotency key"
    input_schema = BaseModel

    def __init__(self) -> None:
        self.keys: list[str] = []

    async def run(self, **kwargs):
        await asyncio.sleep(0)
        self.keys.append(current_tool_call().idempotency_key)
        return "ok"


class TestFanOutIdempotencyKeys:
    @pytest.mark.asyncio
    async def test_same_tool_in_two_branches_gets_different_keys(self):
        tool = _KeyTool()

        def branch() -> NodeFn[MessagesState]:
            return tool_node(tool, input_fn=lambda s: {}, output_fn=lambda s, m: s)

        graph: WorkflowGraph[MessagesState] = WorkflowGraph(MessagesState)
        graph.add_node(
            "fan",
            fan_out(
                [("a", branch()), ("b", branch())],
                lambda state, results: state,
            ),
        )
        graph.set_entry_point("fan")
        graph.add_edge("fan", END)

        async for _ in graph.compile().arun(MessagesState(), run_id="run-1"):
            pass

        assert sorted(tool.keys) == ["run-1:fan:1:a:1", "run-1:fan:1:b:1"]

    @pytest.mark.asyncio
    async def test_nested_fan_outs_join_the_branch_names(self):
        tool = _KeyTool()
        leaf = tool_node(tool, input_fn=lambda s: {}, output_fn=lambda s, m: s)
        inner = fan_out([("x", leaf)], lambda state, results: state)
        outer = fan_out([("o", inner)], lambda state, results: state)

        graph: WorkflowGraph[MessagesState] = WorkflowGraph(MessagesState)
        graph.add_node("fan", outer)
        graph.set_entry_point("fan")
        graph.add_edge("fan", END)

        async for _ in graph.compile().arun(MessagesState(), run_id="run-1"):
            pass

        assert tool.keys == ["run-1:fan:1:o:x:1"]

    @pytest.mark.asyncio
    async def test_outside_a_workflow_run_the_branches_still_run(self):
        tool = _KeyTool()
        leaf = tool_node(tool, input_fn=lambda s: {}, output_fn=lambda s, m: s)

        await fan_out([("a", leaf)], lambda state, results: state)(MessagesState())

        assert len(tool.keys) == 1
