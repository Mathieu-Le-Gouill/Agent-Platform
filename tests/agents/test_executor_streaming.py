import pytest
from pydantic import BaseModel

from agent_platform.agents.agent import Agent
from agent_platform.agents.errors import (
    AgentGuardrailError,
    AgentMaxIterations,
    AgentRecoveryExhausted,
)
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools.base import Tool, ToolStreamChunk
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.schemas.message import UserMessage
from tests.helpers import (
    make_fake_stream,
    make_text_stream_chunks,
    make_tool_call_stream_chunks,
)
from tests.helpers.agents import (
    Answer,
    BlockingOutputGuardrail,
    RewritingGuardrail,
    StrictWeatherTool,
    WeatherTool,
)


class TestExecutorRunStreaming:
    @pytest.mark.asyncio
    async def test_direct_response_yields_text(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["Hello ", "world"])
        )
        agent = Agent(name="test", llm=mock_llm)
        ex = AgentExecutor(agent)
        events = [e async for e in ex.run_streaming("Hi")]
        assert events == ["Hello ", "world"]

    @pytest.mark.asyncio
    async def test_tool_round_then_final_answer(self, mock_llm):
        call_count = 0

        def stream_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_stream(
                    make_tool_call_stream_chunks(
                        [
                            {
                                "id": "c1",
                                "name": "get_weather",
                                "args": {"location": "Paris"},
                            }
                        ]
                    )
                )
            return make_fake_stream(make_text_stream_chunks(["It's sunny in Paris!"]))

        mock_llm.stream.side_effect = stream_side_effect

        registry = ToolRegistry()
        registry.register(WeatherTool())
        agent = Agent(name="weather-agent", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        events = [e async for e in ex.run_streaming("Weather in Paris?")]
        assert events[-1] == "It's sunny in Paris!"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_tool_call_stream_chunks(
                [{"id": "c1", "name": "get_weather", "args": {"location": "Paris"}}]
            )
        )
        registry = ToolRegistry()
        registry.register(WeatherTool())
        agent = Agent(name="loopy", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent, max_iterations=2)

        with pytest.raises(AgentMaxIterations, match="exceeded max iterations"):
            async for _ in ex.run_streaming("Weather?"):
                pass

    @pytest.mark.asyncio
    async def test_repeated_bad_calls_raise_without_yielding_events(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_tool_call_stream_chunks(
                [{"id": "c1", "name": "strict_weather", "args": {}}]
            )
        )
        registry = ToolRegistry()
        registry.register(StrictWeatherTool())
        agent = Agent(name="stuck-streamer", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        events: list = []
        with pytest.raises(AgentRecoveryExhausted):
            async for event in ex.run_streaming("Weather?"):
                events.append(event)

        assert events == []

    @pytest.mark.asyncio
    async def test_blocking_output_guardrail_raises_without_yielding_text(
        self, mock_llm
    ):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["secret ", "text"])
        )
        agent = Agent(
            name="guarded", llm=mock_llm, guardrails=[BlockingOutputGuardrail()]
        )
        ex = AgentExecutor(agent)
        events: list = []

        with pytest.raises(AgentGuardrailError):
            async for event in ex.run_streaming("Hi"):
                events.append(event)

        assert events == []

    @pytest.mark.asyncio
    async def test_rewriting_output_guardrail_yields_the_rewritten_text(self, mock_llm):
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(
            make_text_stream_chunks(["secret ", "text"])
        )
        agent = Agent(name="guarded", llm=mock_llm, guardrails=[RewritingGuardrail()])
        ex = AgentExecutor(agent)

        events = [e async for e in ex.run_streaming("Hi")]

        assert events == ["[redacted]"]

    @pytest.mark.asyncio
    async def test_invalid_response_schema_retry_yields_only_the_accepted_attempt(
        self, mock_llm
    ):
        attempts = [
            make_text_stream_chunks(["not ", "json"]),
            make_text_stream_chunks(['{"value":', " 1}"]),
        ]
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(attempts.pop(0))
        agent = Agent(name="structured", llm=mock_llm, response_schema=Answer)
        ex = AgentExecutor(agent)

        events = [e async for e in ex.run_streaming("Hi")]

        assert events == ['{"value":', " 1}"]

    @pytest.mark.asyncio
    async def test_tool_call_retry_drops_the_first_attempts_events(self, mock_llm):
        attempts = [
            [
                *make_text_stream_chunks(["draft "])[:-1],
                *make_tool_call_stream_chunks(
                    [{"id": "c1", "name": "strict_weather", "args": {}}]
                ),
            ],
            make_tool_call_stream_chunks(
                [
                    {
                        "id": "c2",
                        "name": "strict_weather",
                        "args": {"location": "Paris"},
                    }
                ]
            ),
            make_text_stream_chunks(["Sunny"]),
        ]
        mock_llm.stream.side_effect = lambda **kw: make_fake_stream(attempts.pop(0))
        registry = ToolRegistry()
        registry.register(StrictWeatherTool())
        agent = Agent(name="recovering", llm=mock_llm, tool_registry=registry)
        ex = AgentExecutor(agent)

        events = [e async for e in ex.run_streaming("Weather?")]

        assert "draft " not in events
        assert not any(
            isinstance(e, ToolStreamChunk) and e.is_validation_error for e in events
        )
        assert events[-1] == "Sunny"

    @pytest.mark.asyncio
    async def test_streamed_malformed_arguments_trigger_the_corrective_retry(
        self, mock_llm
    ):
        runs: list[dict] = []

        class _RecordingTool(Tool):
            name = "recorder"
            description = "Records its calls"
            input_schema = BaseModel

            async def run(self, **kwargs):
                runs.append(kwargs)
                return "ok"

        broken = [
            StreamChunk(
                delta="",
                tool_call_deltas=[
                    ToolCallDelta(
                        index=0, id="c1", name="recorder", arguments_delta="{not json"
                    )
                ],
            ),
            StreamChunk(delta="", finish_reason=FinishReason.TOOL_CALL),
        ]
        attempts = [broken, make_text_stream_chunks(["Done"])]
        prompts: list = []

        def stream(**kw):
            prompts.append(list(kw["prompt"].messages))
            return make_fake_stream(attempts.pop(0))

        mock_llm.stream.side_effect = stream
        registry = ToolRegistry()
        registry.register(_RecordingTool())
        ex = AgentExecutor(
            Agent(name="recovering", llm=mock_llm, tool_registry=registry)
        )

        events = [e async for e in ex.run_streaming("Go")]

        assert runs == []
        assert events[-1] == "Done"
        assert len(prompts) == 2
        assert any(
            isinstance(m, UserMessage) and "Invalid JSON arguments" in m.content
            for m in prompts[1]
        )
