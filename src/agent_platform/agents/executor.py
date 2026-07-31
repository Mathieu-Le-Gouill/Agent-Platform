from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from agent_platform.agents.agent import Agent
from agent_platform.agents.errors import AgentMaxIterations, AgentRecoveryExhausted
from agent_platform.agents.tools.base import ToolStreamChunk
from agent_platform.agents.tools.registry import is_tool_validation_error
from agent_platform.agents.validation import retry_once_on_invalid
from agent_platform.core.genai_tracing import GenAIAttributes, traced_operation_span
from agent_platform.core.interfaces.llm.response import ToolCallDelta
from agent_platform.core.schemas.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)

_ThinkActResult = tuple[AssistantMessage, list[ToolMessage]]


def _tool_call_error(result: _ThinkActResult) -> str | None:
    _, tool_messages = result
    bad = [tm for tm in tool_messages if is_tool_validation_error(tm)]
    if not bad:
        return None
    return "; ".join(tm.result.content for tm in bad)


def _assemble_tool_calls(deltas: list[ToolCallDelta]) -> list[ToolCall]:
    """Reassemble fragmented `ToolCallDelta`s (by stream index) into complete `ToolCall`s."""
    assembled: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for delta in deltas:
        if delta.index not in assembled:
            assembled[delta.index] = {"id": "", "name": "", "arguments": ""}
            order.append(delta.index)
        entry = assembled[delta.index]
        if delta.id is not None:
            entry["id"] = delta.id
        if delta.name is not None:
            entry["name"] = delta.name
        if delta.arguments_delta is not None:
            entry["arguments"] += delta.arguments_delta

    calls = []
    for index in order:
        entry = assembled[index]
        try:
            arguments = json.loads(entry["arguments"]) if entry["arguments"] else {}
        except json.JSONDecodeError:
            arguments = {}
        calls.append(ToolCall(id=entry["id"], name=entry["name"], arguments=arguments))
    return calls


class AgentExecutor:
    def __init__(
        self,
        agent: Agent,
        max_iterations: int = 10,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._agent = agent
        self._max_iterations = max_iterations

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    async def run(self, user_input: str, *, conversation_id: str | None = None) -> str:
        messages: list[Message] = [UserMessage(content=user_input)]
        result, _ = await self._execute(messages, conversation_id=conversation_id)
        return result

    async def run_with_messages(
        self, messages: list[Message], *, conversation_id: str | None = None
    ) -> tuple[str, list[Message]]:
        return await self._execute(list(messages), conversation_id=conversation_id)

    async def _think_and_act(self, messages: list[Message]) -> _ThinkActResult:
        assistant_msg = await self._agent.think(messages)
        messages.append(assistant_msg)
        if not assistant_msg.tool_calls:
            return assistant_msg, []
        tool_messages = await self._agent.act(assistant_msg)
        messages.extend(tool_messages)
        return assistant_msg, tool_messages

    async def _correct_tool_call(self, messages: list[Message], error: str) -> None:
        messages.append(
            UserMessage(
                content=(
                    f"Your previous tool call was invalid: {error}. "
                    "Please retry with corrected arguments."
                )
            )
        )

    async def _execute(
        self, messages: list[Message], *, conversation_id: str | None = None
    ) -> tuple[str, list[Message]]:
        with traced_operation_span(
            "invoke_agent",
            {
                GenAIAttributes.AGENT_NAME: self._agent.name,
                GenAIAttributes.CONVERSATION_ID: conversation_id,
            },
        ):
            for _ in range(self._max_iterations):
                try:
                    assistant_msg, _tool_messages = await retry_once_on_invalid(
                        attempt=lambda: self._think_and_act(messages),
                        check=_tool_call_error,
                        correct=lambda error: self._correct_tool_call(messages, error),
                    )
                except AgentRecoveryExhausted as exc:
                    failed_msg, _ = exc.last_result
                    return failed_msg.text, messages

                if not assistant_msg.tool_calls:
                    return assistant_msg.text, messages

            raise AgentMaxIterations(
                f"Agent '{self._agent.name}' exceeded "
                f"max iterations ({self._max_iterations})"
            )

    async def _stream_think_and_act(
        self, messages: list[Message], sink: list[ToolStreamChunk | str]
    ) -> _ThinkActResult:
        text_parts: list[str] = []
        tool_call_deltas: list[ToolCallDelta] = []

        async for chunk in self._agent.think_stream(messages):
            if chunk.delta:
                text_parts.append(chunk.delta)
                sink.append(chunk.delta)
            tool_call_deltas.extend(chunk.tool_call_deltas)

        assistant_msg = AssistantMessage(
            content="".join(text_parts),
            tool_calls=_assemble_tool_calls(tool_call_deltas),
        )
        messages.append(assistant_msg)

        if not assistant_msg.tool_calls:
            return assistant_msg, []

        tool_messages: list[ToolMessage] = []
        async for event in self._agent.act_stream(assistant_msg):
            if isinstance(event, ToolMessage):
                tool_messages.append(event)
                messages.append(event)
            else:
                sink.append(event)
        return assistant_msg, tool_messages

    async def run_streaming(
        self, user_input: str, *, conversation_id: str | None = None
    ) -> AsyncIterator[ToolStreamChunk | str]:
        messages: list[Message] = [UserMessage(content=user_input)]

        with traced_operation_span(
            "invoke_agent",
            {
                GenAIAttributes.AGENT_NAME: self._agent.name,
                GenAIAttributes.CONVERSATION_ID: conversation_id,
            },
        ):
            for _ in range(self._max_iterations):
                sink: list[ToolStreamChunk | str] = []

                try:
                    assistant_msg, _tool_messages = await retry_once_on_invalid(
                        attempt=lambda: self._stream_think_and_act(messages, sink),
                        check=_tool_call_error,
                        correct=lambda error: self._correct_tool_call(messages, error),
                    )
                except AgentRecoveryExhausted:
                    for event in sink:
                        yield event
                    return

                for event in sink:
                    yield event

                if not assistant_msg.tool_calls:
                    return

            raise AgentMaxIterations(
                f"Agent '{self._agent.name}' exceeded "
                f"max iterations ({self._max_iterations})"
            )
