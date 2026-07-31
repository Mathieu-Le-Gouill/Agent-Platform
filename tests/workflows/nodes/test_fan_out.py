import asyncio

import pytest

from agent_platform.core.schemas.message import AssistantMessage
from agent_platform.workflows.nodes.fan_out import fan_out
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
