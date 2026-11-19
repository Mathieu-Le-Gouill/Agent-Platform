import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.schemas.message import (
    AssistantMessage,
    ToolCall,
    UserMessage,
)
from agent_platform.evals.schemas import EvalCase
from agent_platform.evals.targets import (
    agent_executor_target,
    conversation_agent_target,
)
from tests.helpers import make_fake_llm_response


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


class _EchoTool(Tool):
    name = "echo"
    description = "Echoes the input"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return kwargs.get("text", "")


def _user_texts(prompt) -> list[str]:
    return [m.content for m in prompt.messages if isinstance(m, UserMessage)]


class TestConversationAgentTarget:
    @pytest.mark.asyncio
    async def test_extracts_tool_calls_of_the_case(self, mock_llm):
        responses = iter(
            [
                make_fake_llm_response(
                    tool_calls=[{"id": "1", "name": "echo", "args": {"text": "x"}}]
                ),
                make_fake_llm_response(content="final text"),
            ]
        )
        mock_llm.agenerate.side_effect = lambda **_: next(responses)
        registry = ToolRegistry()
        registry.register(_EchoTool())
        agent = ConversationAgent(name="conv", llm=mock_llm, tool_registry=registry)

        output = await conversation_agent_target(agent)(
            EvalCase(id="c1", input="find stuff")
        )

        assert output.text == "final text"
        assert output.tool_calls == ["echo"]

    @pytest.mark.asyncio
    async def test_concurrent_cases_do_not_share_history(self, mock_llm):
        seen: list[list[str]] = []

        async def generate(*, prompt, **_):
            seen.append(_user_texts(prompt))
            # Yield so both cases are in flight before either one finishes.
            await asyncio.sleep(0)
            return make_fake_llm_response(content="done")

        mock_llm.agenerate.side_effect = generate
        agent = ConversationAgent(name="conv", llm=mock_llm)
        target = conversation_agent_target(agent)

        await asyncio.gather(
            target(EvalCase(id="a", input="question a")),
            target(EvalCase(id="b", input="question b")),
        )

        assert sorted(seen) == [["question a"], ["question b"]]
        assert agent.history == []
