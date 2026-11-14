from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

from agent_platform.agents.agent import Agent
from agent_platform.agents.errors import AgentMaxIterations, AgentRecoveryExhausted
from agent_platform.agents.tools.base import ToolStreamChunk
from agent_platform.agents.tools.registry import is_tool_validation_error
from agent_platform.agents.validation import retry_once_on_invalid
from agent_platform.core.interfaces.llm.response import ToolCallDelta
from agent_platform.core.observability import AgentRun, observe
from agent_platform.core.schemas.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)

_ThinkActResult = tuple[AssistantMessage, list[ToolMessage]]

# How a loop stopped: on the model's own final answer, because recovery from
# an invalid tool call failed, or because it ran out of iterations.
EXIT_STOP = "stop"
EXIT_RECOVERY_EXHAUSTED = "recovery_exhausted"
EXIT_MAX_ITERATIONS = "max_iterations"


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


class _LoopTrace:
    """Fills in one `AgentRun` as the loop progresses, and seals its outcome.

    The buffered and streaming paths run the same loop with different
    plumbing, so they share this rather than each filling the observation
    their own way (and drifting).
    """

    def __init__(self, run: AgentRun) -> None:
        self._run = run
        self._finished = False

    def start_iteration(self) -> None:
        self._run.iterations += 1

    def on_retry(self) -> None:
        self._run.recovery_attempted = True

    def record_turn(
        self, assistant_msg: AssistantMessage, tool_messages: list[ToolMessage]
    ) -> None:
        self._run.tool_calls_requested += len(assistant_msg.tool_calls)
        self._run.tool_calls_executed += len(tool_messages)

    def finish(self, exit_reason: str) -> None:
        """Stamp the loop's outcome. Only the first call counts, so an
        explicit reason always wins over the fallback in `_traced_loop`.
        """
        if self._finished:
            return
        self._finished = True
        self._run.exit_reason = exit_reason


@contextmanager
def _traced_loop(
    agent_name: str, conversation_id: str | None, max_iterations: int
) -> Iterator[_LoopTrace]:
    with observe(
        AgentRun(
            agent_name=agent_name,
            conversation_id=conversation_id,
            max_iterations=max_iterations,
        )
    ) as run:
        trace = _LoopTrace(run)
        try:
            yield trace
        except Exception as exc:
            # An unexpected failure still ends the loop, and is worth counting
            # as its own exit reason next to the deliberate ones.
            trace.finish(type(exc).__qualname__)
            raise


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
        with _traced_loop(
            self._agent.name, conversation_id, self._max_iterations
        ) as trace:
            for _ in range(self._max_iterations):
                trace.start_iteration()
                try:
                    assistant_msg, tool_messages = await retry_once_on_invalid(
                        attempt=lambda: self._think_and_act(messages),
                        check=_tool_call_error,
                        correct=lambda error: self._correct_tool_call(messages, error),
                        on_retry=trace.on_retry,
                    )
                except AgentRecoveryExhausted as exc:
                    failed_msg, failed_tool_messages = exc.last_result
                    trace.record_turn(failed_msg, failed_tool_messages)
                    trace.finish(EXIT_RECOVERY_EXHAUSTED)
                    return failed_msg.text, messages

                trace.record_turn(assistant_msg, tool_messages)
                if not assistant_msg.tool_calls:
                    trace.finish(EXIT_STOP)
                    return assistant_msg.text, messages

            trace.finish(EXIT_MAX_ITERATIONS)
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

        with _traced_loop(
            self._agent.name, conversation_id, self._max_iterations
        ) as trace:
            for _ in range(self._max_iterations):
                sink: list[ToolStreamChunk | str] = []
                trace.start_iteration()

                try:
                    assistant_msg, tool_messages = await retry_once_on_invalid(
                        attempt=lambda: self._stream_think_and_act(messages, sink),
                        check=_tool_call_error,
                        correct=lambda error: self._correct_tool_call(messages, error),
                        on_retry=trace.on_retry,
                    )
                except AgentRecoveryExhausted as exc:
                    failed_msg, failed_tool_messages = exc.last_result
                    trace.record_turn(failed_msg, failed_tool_messages)
                    trace.finish(EXIT_RECOVERY_EXHAUSTED)
                    for event in sink:
                        yield event
                    return

                trace.record_turn(assistant_msg, tool_messages)
                for event in sink:
                    yield event

                if not assistant_msg.tool_calls:
                    trace.finish(EXIT_STOP)
                    return

            trace.finish(EXIT_MAX_ITERATIONS)
            raise AgentMaxIterations(
                f"Agent '{self._agent.name}' exceeded "
                f"max iterations ({self._max_iterations})"
            )
