"""The executor's corrective retry: recovering from invalid tool calls,
reusing the calls that already succeeded, and keeping idempotency keys."""

import pytest
from pydantic import BaseModel

from agent_platform.agents.agent import Agent
from agent_platform.agents.errors import AgentRecoveryExhausted
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools import ToolCallValidationError, current_tool_call
from agent_platform.agents.tools.base import Tool, ToolStreamChunk
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.schemas.message import UserMessage
from tests.helpers import (
    make_fake_llm_response,
    make_fake_stream,
    make_text_stream_chunks,
    make_tool_call_stream_chunks,
)
from tests.helpers.agents import Answer, StrictWeatherTool, invoke_agent_span


class TestExecutorToolCallRecovery:
    @pytest.mark.asyncio
    async def test_single_bad_call_recovered_on_retry(self, mock_llm):
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_llm_response(
                    content="",
                    tool_calls=[{"id": "c1", "name": "strict_weather", "args": {}}],
                )
            if call_count == 2:
                return make_fake_llm_response(
                    content="",
                    tool_calls=[
                        {
                            "id": "c2",
                            "name": "strict_weather",
                            "args": {"location": "Paris"},
                        }
                    ],
                )
            return make_fake_llm_response(content="Sunny in Paris")

        mock_llm.agenerate.side_effect = gen
        registry = ToolRegistry()
        registry.register(StrictWeatherTool())
        agent = Agent(name="recovering", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        result = await ex.run("Weather in Paris?")

        assert result == "Sunny in Paris"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_repeated_bad_calls_raise_after_one_retry(self, mock_llm):
        mock_llm.agenerate.return_value = make_fake_llm_response(
            content="", tool_calls=[{"id": "c1", "name": "strict_weather", "args": {}}]
        )
        registry = ToolRegistry()
        registry.register(StrictWeatherTool())
        agent = Agent(name="stuck", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        for run in (
            lambda: ex.run("Weather in Paris?"),
            lambda: ex.run_with_messages([UserMessage(content="Weather in Paris?")]),
        ):
            mock_llm.agenerate.reset_mock()
            with pytest.raises(AgentRecoveryExhausted) as caught:
                await run()

            assert mock_llm.agenerate.await_count == 2
            first, retry = (
                call.kwargs["prompt"].messages
                for call in mock_llm.agenerate.call_args_list
            )
            assert not any("invalid" in str(m.content) for m in first)
            assert any(
                isinstance(m, UserMessage) and "invalid" in m.content for m in retry
            )
            assert caught.value.code == "agent_recovery_exhausted"
            assert caught.value.last_message is not None
            assert caught.value.last_message.tool_calls

    @pytest.mark.asyncio
    async def test_streamed_repeated_bad_calls_raise_without_the_failed_text(
        self, mock_llm
    ):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            [
                *make_text_stream_chunks(["draft "])[:-1],
                *make_tool_call_stream_chunks(
                    [{"id": "c1", "name": "strict_weather", "args": {}}]
                ),
            ]
        )
        registry = ToolRegistry()
        registry.register(StrictWeatherTool())
        ex = AgentExecutor(Agent(name="stuck", llm=mock_llm, tool_registry=registry))
        events: list = []

        with pytest.raises(AgentRecoveryExhausted):
            async for event in ex.run_streaming("Weather in Paris?"):
                events.append(event)

        assert events == []
        assert mock_llm.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_the_schema_recovery_error_of_the_agent_propagates_unchanged(
        self, mock_llm, recorded_spans
    ):
        mock_llm.agenerate.return_value = make_fake_llm_response(content="not json")
        agent = Agent(name="structured", llm=mock_llm, response_schema=Answer)
        ex = AgentExecutor(agent)

        with pytest.raises(AgentRecoveryExhausted) as caught:
            await ex.run("Hi")

        assert caught.value.last_message is not None
        assert caught.value.last_message.text == "not json"
        assert mock_llm.agenerate.await_count == 2
        span = invoke_agent_span(recorded_spans)
        assert span.attributes["agent.exit_reason"] == "recovery_exhausted"

    @pytest.mark.asyncio
    async def test_well_formed_calls_unaffected(self, mock_llm):
        call_count = 0

        async def gen(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_llm_response(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "strict_weather",
                            "args": {"location": "Paris"},
                        }
                    ],
                )
            return make_fake_llm_response(content="Sunny in Paris")

        mock_llm.agenerate.side_effect = gen
        registry = ToolRegistry()
        registry.register(StrictWeatherTool())
        agent = Agent(name="fine", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        result = await ex.run("Weather in Paris?")

        assert result == "Sunny in Paris"
        assert call_count == 2


class _CountingTool(Tool):
    name = "counter"
    description = "Counts how often it ran"
    input_schema = BaseModel

    def __init__(self) -> None:
        self.runs = 0

    async def run(self, **kwargs):
        self.runs += 1
        return {"run": self.runs, "args": kwargs}


class _FlakyTool(Tool):
    name = "flaky"
    description = "Fails on its first run"
    input_schema = BaseModel

    def __init__(self) -> None:
        self.runs = 0

    async def run(self, **kwargs):
        self.runs += 1
        if self.runs == 1:
            raise RuntimeError("boom")
        return "ok"


_VALID_CALL = {"name": "counter", "args": {"item": "a"}}
_INVALID_CALL = {"name": "strict_weather", "args": {}}


def _recovery_registry() -> tuple[ToolRegistry, _CountingTool]:
    counting = _CountingTool()
    registry = ToolRegistry()
    registry.register(counting)
    registry.register(StrictWeatherTool())
    return registry, counting


class TestExecutorCorrectiveRetryReuse:
    @pytest.mark.asyncio
    async def test_a_reemitted_successful_call_does_not_run_again(self, mock_llm):
        responses = [
            make_fake_llm_response(
                content="",
                tool_calls=[{"id": "a1", **_VALID_CALL}, {"id": "a2", **_INVALID_CALL}],
            ),
            make_fake_llm_response(
                content="",
                tool_calls=[
                    {"id": "b1", **_INVALID_CALL, "args": {"location": "Paris"}},
                    {"id": "b2", **_VALID_CALL},
                ],
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        registry, counting = _recovery_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        answer, history = await ex.run_with_messages([UserMessage(content="go")])

        assert answer == "done"
        assert counting.runs == 1
        kept_results = [
            m.result
            for m in history
            if m.role == "tool" and m.result.tool_call_id[0] == "b"
        ]
        assert [r.tool_call_id for r in kept_results] == ["b1", "b2"]
        assert [r.name for r in kept_results] == ["strict_weather", "counter"]
        assert not any(r.is_error for r in kept_results)
        assert '"run": 1' in kept_results[1].content

    @pytest.mark.asyncio
    async def test_a_call_that_failed_runs_again(self, mock_llm):
        flaky_call = {"name": "flaky", "args": {}}
        responses = [
            make_fake_llm_response(
                content="",
                tool_calls=[{"id": "a1", **flaky_call}, {"id": "a2", **_INVALID_CALL}],
            ),
            make_fake_llm_response(
                content="",
                tool_calls=[
                    {"id": "b1", **flaky_call},
                    {"id": "b2", **_INVALID_CALL, "args": {"location": "Paris"}},
                ],
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        flaky = _FlakyTool()
        registry, _ = _recovery_registry()
        registry.register(flaky)
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        await ex.run("go")

        assert flaky.runs == 2

    @pytest.mark.asyncio
    async def test_a_changed_call_runs_again(self, mock_llm):
        responses = [
            make_fake_llm_response(
                content="",
                tool_calls=[{"id": "a1", **_VALID_CALL}, {"id": "a2", **_INVALID_CALL}],
            ),
            make_fake_llm_response(
                content="",
                tool_calls=[
                    {"id": "b1", "name": "counter", "args": {"item": "other"}},
                    {"id": "b2", **_INVALID_CALL, "args": {"location": "Paris"}},
                ],
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        registry, counting = _recovery_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        await ex.run("go")

        assert counting.runs == 2

    @pytest.mark.asyncio
    async def test_each_stored_result_is_reused_at_most_once(self, mock_llm):
        responses = [
            make_fake_llm_response(
                content="",
                tool_calls=[{"id": "a1", **_VALID_CALL}, {"id": "a2", **_INVALID_CALL}],
            ),
            make_fake_llm_response(
                content="",
                tool_calls=[{"id": "b1", **_VALID_CALL}, {"id": "b2", **_VALID_CALL}],
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        registry, counting = _recovery_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        await ex.run("go")

        assert counting.runs == 2

    @pytest.mark.asyncio
    async def test_every_call_reused_runs_no_tool(self, mock_llm):
        responses = [
            make_fake_llm_response(
                content="",
                tool_calls=[{"id": "a1", **_VALID_CALL}, {"id": "a2", **_INVALID_CALL}],
            ),
            make_fake_llm_response(
                content="", tool_calls=[{"id": "b1", **_VALID_CALL}]
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        registry, counting = _recovery_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        answer, history = await ex.run_with_messages([UserMessage(content="go")])

        assert answer == "done"
        assert counting.runs == 1
        (kept,) = [
            m.result
            for m in history
            if m.role == "tool" and m.result.tool_call_id == "b1"
        ]
        assert kept.name == "counter"
        assert not kept.is_error

    @pytest.mark.asyncio
    async def test_a_call_with_malformed_arguments_is_never_reused(self, mock_llm):
        first = make_tool_call_stream_chunks(
            [{"id": "a1", "name": "counter", "args": {}}, {"id": "a2", **_INVALID_CALL}]
        )
        second = [
            StreamChunk(
                delta="",
                tool_call_deltas=[
                    ToolCallDelta(
                        index=0, id="b1", name="counter", arguments_delta="{not json"
                    )
                ],
            ),
            StreamChunk(delta="", finish_reason=FinishReason.TOOL_CALL),
        ]
        attempts = [first, second, make_text_stream_chunks(["done"])]
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(attempts.pop(0))
        registry, counting = _recovery_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        with pytest.raises(AgentRecoveryExhausted):
            [e async for e in ex.run_streaming("go")]

        assert counting.runs == 1

    @pytest.mark.asyncio
    async def test_streaming_reuses_the_call_and_emits_its_final_chunk(self, mock_llm):
        attempts = [
            make_tool_call_stream_chunks(
                [{"id": "a1", **_VALID_CALL}, {"id": "a2", **_INVALID_CALL}]
            ),
            make_tool_call_stream_chunks(
                [
                    {"id": "b1", **_INVALID_CALL, "args": {"location": "Paris"}},
                    {"id": "b2", **_VALID_CALL},
                ]
            ),
            make_text_stream_chunks(["done"]),
        ]
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(attempts.pop(0))
        registry, counting = _recovery_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        events = [e async for e in ex.run_streaming("go")]

        assert counting.runs == 1
        finals = [e for e in events if isinstance(e, ToolStreamChunk) and e.is_final]
        assert [f.tool_call_id for f in finals] == ["b1", "b2"]
        assert '"run": 1' in finals[1].delta
        assert not any(f.is_error for f in finals)
        assert events[-1] == "done"

    @pytest.mark.asyncio
    async def test_the_trace_counts_the_reused_call(self, mock_llm, recorded_spans):
        responses = [
            make_fake_llm_response(
                content="",
                tool_calls=[{"id": "a1", **_VALID_CALL}, {"id": "a2", **_INVALID_CALL}],
            ),
            make_fake_llm_response(
                content="",
                tool_calls=[
                    {"id": "b1", **_VALID_CALL},
                    {"id": "b2", **_INVALID_CALL, "args": {"location": "Paris"}},
                ],
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        registry, _ = _recovery_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        await ex.run("go")

        span = invoke_agent_span(recorded_spans)
        assert span.attributes["agent.tool_calls_requested"] == 2
        assert span.attributes["agent.tool_calls_executed"] == 2


class _KeyedTool(Tool):
    """Records the idempotency key of each run; its first `failures` runs raise."""

    name = "keyed"
    description = "Records its idempotency key"
    input_schema = BaseModel

    def __init__(self, failures: int = 1) -> None:
        self.keys: list[str | None] = []
        self._failures = failures

    async def run(self, **kwargs):
        context = current_tool_call()
        self.keys.append(context.idempotency_key if context else None)
        if len(self.keys) <= self._failures:
            raise RuntimeError("timed out after the side effect started")
        return "ok"


def _keyed_registry(failures: int = 1) -> tuple[ToolRegistry, _KeyedTool]:
    keyed = _KeyedTool(failures)
    registry, _ = _recovery_registry()
    registry.register(keyed)
    return registry, keyed


_KEYED_CALL = {"name": "keyed", "args": {"to": "bob"}}
_CORRECTED_CALL = {**_INVALID_CALL, "args": {"location": "Paris"}}


class TestExecutorCorrectiveRetryKeys:
    @pytest.mark.asyncio
    async def test_a_failed_call_reemitted_unchanged_keeps_its_first_key(
        self, mock_llm
    ):
        responses = [
            make_fake_llm_response(
                content="",
                tool_calls=[{"id": "a1", **_KEYED_CALL}, {"id": "a2", **_INVALID_CALL}],
            ),
            make_fake_llm_response(
                content="",
                tool_calls=[
                    {"id": "b1", **_KEYED_CALL},
                    {"id": "b2", **_CORRECTED_CALL},
                ],
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        registry, keyed = _keyed_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        answer, history = await ex.run_with_messages([UserMessage(content="go")])

        assert answer == "done"
        assert keyed.keys == ["a1", "a1"]
        retried = [
            m.result
            for m in history
            if m.role == "tool" and m.result.name == "keyed" and not m.result.is_error
        ]
        assert [r.tool_call_id for r in retried] == ["b1"]
        assert current_tool_call() is None

    @pytest.mark.asyncio
    async def test_a_call_that_failed_validation_reemitted_unchanged_keeps_its_key(
        self, mock_llm
    ):
        class _RejectingOnceTool(_KeyedTool):
            async def run(self, **kwargs):
                context = current_tool_call()
                self.keys.append(context.idempotency_key if context else None)
                if len(self.keys) == 1:
                    raise ToolCallValidationError("bad arguments")
                return "ok"

        rejecting = _RejectingOnceTool()
        registry, _ = _recovery_registry()
        registry.register(rejecting)
        responses = [
            make_fake_llm_response(
                content="", tool_calls=[{"id": "a1", **_KEYED_CALL}]
            ),
            make_fake_llm_response(
                content="", tool_calls=[{"id": "b1", **_KEYED_CALL}]
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        answer = await ex.run("go")

        assert answer == "done"
        assert rejecting.keys == ["a1", "a1"]

    @pytest.mark.asyncio
    async def test_a_malformed_call_does_not_take_a_failed_calls_key(self, mock_llm):
        first = make_tool_call_stream_chunks(
            [
                {"id": "a1", "name": "keyed", "args": {}},
                {"id": "a2", **_INVALID_CALL},
            ]
        )
        second = [
            StreamChunk(
                delta="",
                tool_call_deltas=[
                    ToolCallDelta(
                        index=0, id="b1", name="keyed", arguments_delta="{not json"
                    ),
                    ToolCallDelta(index=1, id="b2", name="keyed", arguments_delta="{}"),
                    ToolCallDelta(
                        index=2,
                        id="b3",
                        name="strict_weather",
                        arguments_delta='{"location": "Paris"}',
                    ),
                ],
            ),
            StreamChunk(delta="", finish_reason=FinishReason.TOOL_CALL),
        ]
        attempts = [first, second, make_text_stream_chunks(["done"])]
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(attempts.pop(0))
        registry, keyed = _keyed_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        with pytest.raises(AgentRecoveryExhausted):
            [e async for e in ex.run_streaming("go")]

        assert keyed.keys[-1] == "a1"

    @pytest.mark.asyncio
    async def test_streaming_keeps_the_first_key_of_a_failed_call(self, mock_llm):
        attempts = [
            make_tool_call_stream_chunks(
                [{"id": "a1", **_KEYED_CALL}, {"id": "a2", **_INVALID_CALL}]
            ),
            make_tool_call_stream_chunks(
                [{"id": "b1", **_KEYED_CALL}, {"id": "b2", **_CORRECTED_CALL}]
            ),
            make_text_stream_chunks(["done"]),
        ]
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(attempts.pop(0))
        registry, keyed = _keyed_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        events = [e async for e in ex.run_streaming("go")]

        assert events[-1] == "done"
        assert keyed.keys == ["a1", "a1"]
        finals = [e for e in events if isinstance(e, ToolStreamChunk) and e.is_final]
        assert "b1" in [f.tool_call_id for f in finals]
        assert current_tool_call() is None

    @pytest.mark.asyncio
    async def test_a_changed_or_new_call_gets_its_own_key(self, mock_llm):
        responses = [
            make_fake_llm_response(
                content="",
                tool_calls=[{"id": "a1", **_KEYED_CALL}, {"id": "a2", **_INVALID_CALL}],
            ),
            make_fake_llm_response(
                content="",
                tool_calls=[
                    {"id": "b1", "name": "keyed", "args": {"to": "alice"}},
                    {"id": "b2", **_CORRECTED_CALL},
                    {"id": "b3", "name": "keyed", "args": {"to": "carol"}},
                ],
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        registry, keyed = _keyed_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        await ex.run("go")

        assert keyed.keys[0] == "a1"
        assert sorted(keyed.keys[1:]) == ["b1", "b3"]

    @pytest.mark.asyncio
    async def test_identical_failed_calls_map_one_to_one_to_their_keys(self, mock_llm):
        responses = [
            make_fake_llm_response(
                content="",
                tool_calls=[
                    {"id": "a1", **_KEYED_CALL},
                    {"id": "a2", **_KEYED_CALL},
                    {"id": "a3", **_INVALID_CALL},
                ],
            ),
            make_fake_llm_response(
                content="",
                tool_calls=[
                    {"id": "b1", **_KEYED_CALL},
                    {"id": "b2", **_KEYED_CALL},
                    {"id": "b3", **_CORRECTED_CALL},
                ],
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        registry, keyed = _keyed_registry(failures=2)
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        await ex.run("go")

        assert sorted(keyed.keys[:2]) == ["a1", "a2"]
        assert sorted(keyed.keys[2:]) == ["a1", "a2"]

    @pytest.mark.asyncio
    async def test_a_new_call_owning_the_failed_id_does_not_share_its_key(
        self, mock_llm
    ):
        responses = [
            make_fake_llm_response(
                content="",
                tool_calls=[{"id": "a1", **_KEYED_CALL}, {"id": "a2", **_INVALID_CALL}],
            ),
            make_fake_llm_response(
                content="",
                tool_calls=[
                    {"id": "a1", "name": "keyed", "args": {"to": "carol"}},
                    {"id": "b2", **_KEYED_CALL},
                    {"id": "b3", **_CORRECTED_CALL},
                ],
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        registry, keyed = _keyed_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        assert await ex.run("go") == "done"

        assert sorted(keyed.keys[1:]) == ["a1", "b2"]

    @pytest.mark.asyncio
    async def test_streaming_a_new_call_owning_the_failed_id_does_not_share_its_key(
        self, mock_llm
    ):
        attempts = [
            make_tool_call_stream_chunks(
                [{"id": "a1", **_KEYED_CALL}, {"id": "a2", **_INVALID_CALL}]
            ),
            make_tool_call_stream_chunks(
                [
                    {"id": "a1", "name": "keyed", "args": {"to": "carol"}},
                    {"id": "b2", **_KEYED_CALL},
                    {"id": "b3", **_CORRECTED_CALL},
                ]
            ),
            make_text_stream_chunks(["done"]),
        ]
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(attempts.pop(0))
        registry, keyed = _keyed_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        events = [e async for e in ex.run_streaming("go")]

        assert events[-1] == "done"
        assert keyed.keys[1:] == ["a1", "b2"]

    @pytest.mark.asyncio
    async def test_a_malformed_call_leaves_no_key_for_the_retry(self, mock_llm):
        first = [
            StreamChunk(
                delta="",
                tool_call_deltas=[
                    ToolCallDelta(
                        index=0, id="a1", name="keyed", arguments_delta="{not json"
                    ),
                    ToolCallDelta(index=1, id="a2", name="keyed", arguments_delta="{}"),
                ],
            ),
            StreamChunk(delta="", finish_reason=FinishReason.TOOL_CALL),
        ]
        attempts = [
            first,
            make_tool_call_stream_chunks([{"id": "b1", "name": "keyed", "args": {}}]),
            make_text_stream_chunks(["done"]),
        ]
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(attempts.pop(0))
        registry, keyed = _keyed_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        events = [e async for e in ex.run_streaming("go")]

        assert events[-1] == "done"
        assert keyed.keys == ["a2", "a2"]

    @pytest.mark.asyncio
    async def test_outside_the_corrective_retry_keys_are_the_call_ids(self, mock_llm):
        responses = [
            make_fake_llm_response(
                content="", tool_calls=[{"id": "a1", **_KEYED_CALL}]
            ),
            make_fake_llm_response(
                content="", tool_calls=[{"id": "c1", **_KEYED_CALL}]
            ),
            make_fake_llm_response(content="done"),
        ]
        mock_llm.agenerate.side_effect = lambda **kw: responses.pop(0)
        registry, keyed = _keyed_registry()
        ex = AgentExecutor(Agent(name="a", llm=mock_llm, tool_registry=registry))

        await ex.run("go")

        assert keyed.keys == ["a1", "c1"]
