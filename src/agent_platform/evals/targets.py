from __future__ import annotations

from collections.abc import Awaitable, Callable

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
    async def _target(case: EvalCase) -> EvalOutput:
        turns_before = len(agent.history)
        text = await agent.chat(case.input)
        new_messages = agent.history[turns_before:]
        tool_calls = [
            message.result.name
            for message in new_messages
            if isinstance(message, ToolMessage)
        ]
        return EvalOutput(text=text, tool_calls=tool_calls)

    return _target


__all__ = ["EvalTarget", "agent_executor_target", "conversation_agent_target"]
