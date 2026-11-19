from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import uuid4

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.agents.executor import AgentExecutor
from agent_platform.core.schemas.message import (
    AssistantMessage,
    ToolMessage,
    UserMessage,
)
from agent_platform.evals.schemas import EvalCase, EvalOutput

EvalTarget = Callable[[EvalCase], Awaitable[EvalOutput]]


def agent_executor_target(executor: AgentExecutor) -> EvalTarget:
    async def _target(case: EvalCase) -> EvalOutput:
        text, messages = await executor.run_with_messages(
            [UserMessage(content=case.input)]
        )
        tool_calls = [
            tc.name
            for message in messages
            if isinstance(message, AssistantMessage)
            for tc in message.tool_calls
        ]
        return EvalOutput(text=text, tool_calls=tool_calls)

    return _target


def conversation_agent_target(agent: ConversationAgent) -> EvalTarget:
    """Returns a target that runs each case on its own fork of `agent`.

    A fork starts with an empty history, so cases run concurrently by
    `EvalRunner` never see each other's turns, and `agent` itself is untouched.
    """

    async def _target(case: EvalCase) -> EvalOutput:
        case_agent = agent.with_conversation(uuid4())
        text = await case_agent.chat(case.input)
        tool_calls = [
            message.result.name
            for message in case_agent.history
            if isinstance(message, ToolMessage)
        ]
        return EvalOutput(text=text, tool_calls=tool_calls)

    return _target


__all__ = ["EvalTarget", "agent_executor_target", "conversation_agent_target"]
