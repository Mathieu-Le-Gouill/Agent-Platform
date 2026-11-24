from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

from agent_platform.agents.agent import Agent
from agent_platform.agents.errors import AgentMaxIterations, AgentRecoveryExhausted
from agent_platform.agents.tools.base import ToolStreamChunk
from agent_platform.agents.tools.call_context import bind_idempotency_keys
from agent_platform.agents.tools.registry import is_tool_validation_error
from agent_platform.core.errors import RecoveryExhausted
from agent_platform.core.interfaces.llm.structured import StructuredOutputError
from agent_platform.core.observability import AgentRun, observe
from agent_platform.core.retry import retry_once_on_invalid
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


def _call_signature(call: ToolCall) -> tuple[str, str]:
    return call.name, json.dumps(call.arguments, sort_keys=True, default=str)


class _PreviousAttempt:
    """The tool calls of the attempt before the corrective retry.

    When the model re-emits a call with the same name and arguments after a
    correction, the retry must not look like a new intent to a tool with a
    side effect: a call that succeeded has its result reused instead of
    running again, and a call that failed runs again under the failed call's
    idempotency key, so the tool can recognise work it may have started.
    Each stored result and each failed call's key is used at most once.
    """

    def __init__(self) -> None:
        self._results: list[tuple[tuple[str, str], ToolMessage]] = []
        self._failed: list[tuple[tuple[str, str], str]] = []

    def take(self, call: ToolCall) -> ToolMessage | None:
        """Return the stored result for `call`, re-labelled with its id.

        A call whose arguments did not parse never matches: its `arguments`
        are an empty fallback, and reusing a result would hide the error.
        """
        if call.arguments_error is not None:
            return None
        signature = _call_signature(call)
        for index, (stored_signature, message) in enumerate(self._results):
            if stored_signature == signature:
                del self._results[index]
                return ToolMessage(
                    result=message.result.model_copy(update={"tool_call_id": call.id}),
                    metadata=dict(message.metadata),
                )
        return None

    def keys_for(self, calls: list[ToolCall]) -> dict[str, str]:
        """Map the id of each call in `calls` that repeats a failed call to
        that failed call's id, the idempotency key it must run under."""
        keys: dict[str, str] = {}
        for call in calls:
            if call.arguments_error is not None:
                continue
            signature = _call_signature(call)
            for index, (failed_signature, failed_id) in enumerate(self._failed):
                if failed_signature == signature:
                    del self._failed[index]
                    keys[call.id] = failed_id
                    break
        return keys

    def remember(
        self, assistant_msg: AssistantMessage, tool_messages: list[ToolMessage]
    ) -> None:
        """Replace what is stored with the outcome of the attempt just run."""
        pairs = list(zip(assistant_msg.tool_calls, tool_messages))
        self._results = [
            (_call_signature(call), message)
            for call, message in pairs
            if not message.result.is_error
        ]
        self._failed = [
            (_call_signature(call), call.id)
            for call, message in pairs
            if message.result.is_error and call.arguments_error is None
        ]


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
        except AgentRecoveryExhausted:
            # The agent's schema retry raised it inside an attempt: same exit
            # reason as the tool-call retry the executor handles itself.
            trace.finish(EXIT_RECOVERY_EXHAUSTED)
            raise
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

    async def run_in_place(
        self, messages: list[Message], *, conversation_id: str | None = None
    ) -> str:
        """Run the loop on `messages` itself and return the final answer text.

        Each message of the turn is appended to `messages` as it happens and
        stays there when the loop raises, so a caller can keep the tool calls
        that already ran.
        """
        result, _ = await self._execute(messages, conversation_id=conversation_id)
        return result

    async def _think_and_act(
        self, messages: list[Message], previous: _PreviousAttempt
    ) -> _ThinkActResult:
        assistant_msg = await self._agent.think(messages)
        messages.append(assistant_msg)
        if not assistant_msg.tool_calls:
            return assistant_msg, []
        tool_messages = await self._act(assistant_msg, previous)
        messages.extend(tool_messages)
        previous.remember(assistant_msg, tool_messages)
        return assistant_msg, tool_messages

    async def _act(
        self, assistant_msg: AssistantMessage, previous: _PreviousAttempt
    ) -> list[ToolMessage]:
        """Run the calls of `assistant_msg`, except those that already succeeded
        in the attempt before the corrective retry, repeated failed calls under
        their earlier key; results keep the call order."""
        reused = [previous.take(call) for call in assistant_msg.tool_calls]
        pending = [
            call
            for call, stored in zip(assistant_msg.tool_calls, reused)
            if stored is None
        ]
        with bind_idempotency_keys(previous.keys_for(pending)):
            if len(pending) == len(reused):
                return await self._agent.act(assistant_msg)
            fresh = iter(
                await self._agent.act(
                    assistant_msg.model_copy(update={"tool_calls": pending})
                )
                if pending
                else []
            )
        return [stored if stored is not None else next(fresh) for stored in reused]

    def _recovery_exhausted(
        self, last_message: AssistantMessage
    ) -> AgentRecoveryExhausted:
        return AgentRecoveryExhausted(
            f"Agent '{self._agent.name}' could not produce a valid tool call",
            last_message=last_message,
        )

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
                previous = _PreviousAttempt()
                try:
                    assistant_msg, tool_messages = await retry_once_on_invalid(
                        attempt=lambda: self._think_and_act(messages, previous),
                        check=_tool_call_error,
                        correct=lambda error: self._correct_tool_call(messages, error),
                        on_retry=trace.on_retry,
                    )
                except StructuredOutputError:
                    # Raised by a structured call inside the attempt, not by
                    # this retry: its last_result is not a turn to record.
                    raise
                except RecoveryExhausted as exc:
                    failed_msg, failed_tool_messages = exc.last_result
                    trace.record_turn(failed_msg, failed_tool_messages)
                    trace.finish(EXIT_RECOVERY_EXHAUSTED)
                    raise self._recovery_exhausted(failed_msg) from exc

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
        self,
        messages: list[Message],
        sink: list[ToolStreamChunk | str],
        previous: _PreviousAttempt,
    ) -> _ThinkActResult:
        assistant_msg, deltas = await self._agent.think_from_stream(messages)
        sink.extend(deltas)
        messages.append(assistant_msg)

        if not assistant_msg.tool_calls:
            return assistant_msg, []

        tool_messages: list[ToolMessage] = []
        for call in assistant_msg.tool_calls:
            stored = previous.take(call)
            if stored is not None:
                sink.append(
                    ToolStreamChunk(
                        tool_call_id=call.id,
                        delta=stored.result.content,
                        is_final=True,
                    )
                )
                tool_messages.append(stored)
                messages.append(stored)
                continue
            # Bound here, in this coroutine, and not around a yield to the
            # caller, so the reset runs in the context that set it.
            with bind_idempotency_keys(previous.keys_for([call])):
                async for event in self._agent.act_stream(
                    assistant_msg.model_copy(update={"tool_calls": [call]})
                ):
                    if isinstance(event, ToolMessage):
                        tool_messages.append(event)
                        messages.append(event)
                    else:
                        sink.append(event)
        previous.remember(assistant_msg, tool_messages)
        return assistant_msg, tool_messages

    async def run_streaming(
        self, user_input: str, *, conversation_id: str | None = None
    ) -> AsyncIterator[ToolStreamChunk | str]:
        messages: list[Message] = [UserMessage(content=user_input)]

        with _traced_loop(
            self._agent.name, conversation_id, self._max_iterations
        ) as trace:
            for _ in range(self._max_iterations):
                # Each attempt fills its own buffer, so only the attempt kept
                # after the corrective retry reaches the client.
                sink: list[ToolStreamChunk | str] = []
                previous = _PreviousAttempt()
                trace.start_iteration()

                async def attempt() -> _ThinkActResult:
                    nonlocal sink
                    sink = []
                    return await self._stream_think_and_act(messages, sink, previous)

                try:
                    assistant_msg, tool_messages = await retry_once_on_invalid(
                        attempt=attempt,
                        check=_tool_call_error,
                        correct=lambda error: self._correct_tool_call(messages, error),
                        on_retry=trace.on_retry,
                    )
                except StructuredOutputError:
                    # Raised by a structured call inside the attempt, not by
                    # this retry: its last_result is not a turn to record.
                    raise
                except RecoveryExhausted as exc:
                    failed_msg, failed_tool_messages = exc.last_result
                    trace.record_turn(failed_msg, failed_tool_messages)
                    trace.finish(EXIT_RECOVERY_EXHAUSTED)
                    raise self._recovery_exhausted(failed_msg) from exc

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
