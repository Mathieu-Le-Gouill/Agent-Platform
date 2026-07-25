from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_platform.core.schemas.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    ToolResult,
)
from agent_platform.evals.schemas import EvalCase
from agent_platform.evals.targets import (
    agent_executor_target,
    conversation_agent_target,
)


class _FakeConversationAgent:
    def __init__(self, history_after: list) -> None:
        self._history: list = []
        self._history_after = history_after
        self.chat_called_with: str | None = None

    @property
    def history(self) -> list:
        return list(self._history)

    async def chat(self, user_input: str) -> str:
        self.chat_called_with = user_input
        self._history = self._history_after
        return "final text"


class TestAgentExecutorTarget:
    @pytest.mark.asyncio
    async def test_extracts_text_and_tool_calls(self):
        executor = MagicMock()
        executor.run_with_messages = AsyncMock(
            return_value=(
                "final answer",
                [
                    AssistantMessage(
                        content="",
                        tool_calls=[ToolCall(id="1", name="search", arguments={})],
                    )
                ],
            )
        )

        target = agent_executor_target(executor)
        output = await target(EvalCase(id="c1", input="find stuff"))

        assert output.text == "final answer"
        assert output.tool_calls == ["search"]
        args, _ = executor.run_with_messages.call_args
        assert args[0][0].content == "find stuff"

    @pytest.mark.asyncio
    async def test_no_tool_calls(self):
        executor = MagicMock()
        executor.run_with_messages = AsyncMock(
            return_value=("just text", [AssistantMessage(content="just text")])
        )

        target = agent_executor_target(executor)
        output = await target(EvalCase(id="c1", input="hi"))

        assert output.tool_calls == []


class TestConversationAgentTarget:
    @pytest.mark.asyncio
    async def test_extracts_new_tool_calls_from_history_diff(self):
        history_after = [
            ToolMessage(result=ToolResult(tool_call_id="1", name="search", content="x"))
        ]
        fake_agent = _FakeConversationAgent(history_after)
        target = conversation_agent_target(fake_agent)

        output = await target(EvalCase(id="c1", input="find stuff"))

        assert output.text == "final text"
        assert output.tool_calls == ["search"]
        assert fake_agent.chat_called_with == "find stuff"
