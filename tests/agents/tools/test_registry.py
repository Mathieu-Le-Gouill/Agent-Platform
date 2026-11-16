import asyncio
import json
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from pydantic import BaseModel

from agent_platform.agents.tools import (
    Tool,
    ToolCallValidationError,
    ToolError,
    ToolRegistry,
    is_tool_validation_error,
)
from agent_platform.agents.tools.errors import ToolTimeoutError
from agent_platform.core.schemas.document import ImageDocument
from agent_platform.core.schemas.enums import ImageFormat
from agent_platform.core.schemas.message import ToolCall


class _DummyTool(Tool):
    name = "dummy"
    description = "A dummy tool"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return {"handled": True}


class _StrictInput(BaseModel):
    location: str


class _StrictTool(Tool):
    name = "strict"
    description = "A tool requiring a specific schema"
    input_schema = _StrictInput

    async def run(self, **kwargs):
        return f"weather for {kwargs['location']}"


class _OtherTool(Tool):
    name = "other"
    description = "Another tool"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return None


class _StreamingTool(Tool):
    name = "streamer"
    description = "A tool that streams output"
    input_schema = BaseModel
    supports_streaming = True

    async def astream(self, **kwargs):
        yield "hello "
        yield "world"


class _FailingStreamingTool(Tool):
    name = "failing_streamer"
    description = "A streaming tool that raises mid-stream"
    input_schema = BaseModel
    supports_streaming = True

    async def astream(self, **kwargs):
        yield "partial"
        raise RuntimeError("stream broke")


class _CancelledStreamingTool(Tool):
    name = "cancelled_streamer"
    description = "A streaming tool that gets cancelled"
    input_schema = BaseModel
    supports_streaming = True

    async def astream(self, **kwargs):
        raise asyncio.CancelledError()
        yield  # pragma: no cover


class _StrictStreamingInput(BaseModel):
    location: str


class _StrictStreamingTool(Tool):
    name = "strict_streamer"
    description = "A streaming tool requiring a specific schema"
    input_schema = _StrictStreamingInput
    supports_streaming = True

    async def astream(self, **kwargs):
        yield kwargs[
            "location"
        ]  # pragma: no cover - should never run when args are bad


@pytest.fixture
def registry():
    return ToolRegistry()


class TestToolRegistry:
    def test_register_adds_tool(self, registry):
        tool = _DummyTool()
        registry.register(tool)
        assert registry.get("dummy") is tool

    def test_register_duplicate_raises(self, registry):
        registry.register(_DummyTool())
        with pytest.raises(ToolError, match="already registered"):
            registry.register(_DummyTool())

    def test_get_unknown_raises(self, registry):
        with pytest.raises(ToolError, match="Unknown tool"):
            registry.get("nonexistent")

    def test_get_returns_correct_tool(self, registry):
        dummy = _DummyTool()
        other = _OtherTool()
        registry.register(dummy)
        registry.register(other)
        assert registry.get("dummy") is dummy
        assert registry.get("other") is other

    def test_all_returns_copy(self, registry):
        dummy = _DummyTool()
        registry.register(dummy)
        result = registry.all()
        assert result == {"dummy": dummy}
        result["extra"] = None
        assert "extra" not in registry.all()

    def test_remove_existing(self, registry):
        registry.register(_DummyTool())
        registry.remove("dummy")
        with pytest.raises(ToolError, match="Unknown tool"):
            registry.get("dummy")

    def test_remove_unknown_raises(self, registry):
        with pytest.raises(ToolError, match="Unknown tool"):
            registry.remove("nonexistent")

    def test_len(self, registry):
        assert len(registry) == 0
        registry.register(_DummyTool())
        assert len(registry) == 1
        registry.register(_OtherTool())
        assert len(registry) == 2

    def test_contains(self, registry):
        assert "dummy" not in registry
        registry.register(_DummyTool())
        assert "dummy" in registry
        assert "other" not in registry

    def test_iteration(self, registry):
        dummy = _DummyTool()
        other = _OtherTool()
        registry.register(dummy)
        registry.register(other)
        tools = list(registry)
        assert dummy in tools
        assert other in tools
        assert len(tools) == 2

    def test_register_twice_same_instance_raises(self, registry):
        tool = _DummyTool()
        registry.register(tool)
        with pytest.raises(ToolError, match="already registered"):
            registry.register(tool)

    @pytest.mark.asyncio
    async def test_resolve_call_lookup_and_execute(self, registry):
        registry.register(_DummyTool())
        call = ToolCall(id="call_1", name="dummy", arguments={})
        result = await registry.resolve_call(call)
        assert result == {"handled": True}

    @pytest.mark.asyncio
    async def test_resolve_call_unknown_tool(self, registry):
        call = ToolCall(id="call_1", name="nonexistent", arguments={})
        with pytest.raises(ToolError, match="Unknown tool"):
            await registry.resolve_call(call)

    @pytest.mark.asyncio
    async def test_resolve_call_passes_arguments(self, registry):
        tool = _DummyTool()
        tool.run = AsyncMock(return_value="done")  # type: ignore[method-assign]
        registry.register(tool)
        call = ToolCall(id="call_2", name="dummy", arguments={"key": "val"})
        await registry.resolve_call(call)
        tool.run.assert_awaited_once_with(key="val")  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_resolve_call_validates_arguments(self, registry):
        tool = _StrictTool()
        registry.register(tool)
        call = ToolCall(id="call_3", name="strict", arguments={"location": "Paris"})
        result = await registry.resolve_call(call)
        assert result == "weather for Paris"

    @pytest.mark.asyncio
    async def test_resolve_call_invalid_arguments_raises_before_run(self, registry):
        tool = _StrictTool()
        tool.run = AsyncMock()  # type: ignore[method-assign]
        registry.register(tool)
        call = ToolCall(id="call_4", name="strict", arguments={})
        with pytest.raises(ToolCallValidationError, match="Invalid arguments"):
            await registry.resolve_call(call)
        tool.run.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_call_and_wrap_surfaces_validation_error(self, registry):
        registry.register(_StrictTool())
        call = ToolCall(id="call_5", name="strict", arguments={})
        message = await registry.call_and_wrap(call)
        assert message.result.is_error is True
        assert "Invalid arguments" in message.result.content

    @pytest.mark.asyncio
    async def test_call_and_wrap_tags_validation_error(self, registry):
        registry.register(_StrictTool())
        call = ToolCall(id="call_6", name="strict", arguments={})
        message = await registry.call_and_wrap(call)
        assert is_tool_validation_error(message) is True

    @pytest.mark.asyncio
    async def test_call_and_wrap_does_not_tag_other_errors(self, registry):
        tool = _DummyTool()
        tool.run = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        registry.register(tool)
        call = ToolCall(id="call_7", name="dummy", arguments={})
        message = await registry.call_and_wrap(call)
        assert message.result.is_error is True
        assert is_tool_validation_error(message) is False

    @pytest.mark.asyncio
    async def test_call_and_wrap_does_not_tag_success(self, registry):
        registry.register(_DummyTool())
        call = ToolCall(id="call_8", name="dummy", arguments={})
        message = await registry.call_and_wrap(call)
        assert is_tool_validation_error(message) is False


class TestCallAndWrap:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self, registry):
        tool = _DummyTool()
        tool.run = AsyncMock(side_effect=asyncio.CancelledError())  # type: ignore[method-assign]
        registry.register(tool)
        call = ToolCall(id="c1", name="dummy", arguments={})
        with pytest.raises(asyncio.CancelledError):
            await registry.call_and_wrap(call)


class _ReturnTool(Tool):
    name = "returner"
    description = "Returns whatever it was given"
    input_schema = BaseModel

    def __init__(self, value):
        self._value = value

    async def run(self, **kwargs):
        return self._value


class _Point(BaseModel):
    x: int
    y: int


async def _wrap(value, **registry_kwargs):
    registry = ToolRegistry(**registry_kwargs)
    registry.register(_ReturnTool(value))
    return await registry.call_and_wrap(
        ToolCall(id="c1", name="returner", arguments={})
    )


class TestCallAndWrapSerialization:
    @pytest.mark.asyncio
    async def test_none_gives_empty_content(self):
        message = await _wrap(None)
        assert message.result.content == ""

    @pytest.mark.asyncio
    async def test_dict_is_json(self):
        message = await _wrap({"handled": True})
        assert message.result.content == '{"handled": true}'

    @pytest.mark.asyncio
    async def test_non_ascii_text_is_kept_readable(self):
        message = await _wrap({"text": "déjà vu 東京"})
        assert message.result.content == '{"text": "déjà vu 東京"}'

    @pytest.mark.asyncio
    async def test_image_document_replaces_bytes_with_a_placeholder(self):
        image = ImageDocument(content=b"\x89PNG" * 10, format=ImageFormat.PNG)

        message = await _wrap(image)

        parsed = json.loads(message.result.content)
        assert parsed["content"] == "<40 bytes>"
        assert "PNG" not in message.result.content

    @pytest.mark.asyncio
    async def test_list_of_models_is_a_json_array(self):
        message = await _wrap([_Point(x=1, y=2), _Point(x=3, y=4)])
        assert json.loads(message.result.content) == [
            {"x": 1, "y": 2},
            {"x": 3, "y": 4},
        ]

    @pytest.mark.asyncio
    async def test_tuple_with_bytes_is_walked(self):
        message = await _wrap((b"abc", "ok"))
        assert json.loads(message.result.content) == ["<3 bytes>", "ok"]

    @pytest.mark.asyncio
    async def test_unencodable_value_falls_back_to_str(self):
        message = await _wrap({"id": UUID(int=1)})
        assert json.loads(message.result.content) == {"id": str(UUID(int=1))}

    @pytest.mark.asyncio
    async def test_short_string_is_unchanged(self):
        message = await _wrap("plain text")
        assert message.result.content == "plain text"

    @pytest.mark.asyncio
    async def test_long_string_is_capped_with_a_marker(self):
        message = await _wrap("a" * 50_000)

        content = message.result.content
        assert content.startswith("a" * 20_000)
        assert "truncated" in content
        assert "50000" in content
        assert len(content) < 20_200

    @pytest.mark.asyncio
    async def test_custom_cap(self):
        message = await _wrap("a" * 100, max_result_chars=10)
        assert message.result.content.startswith("a" * 10 + "\n")
        assert "a" * 11 not in message.result.content

    @pytest.mark.asyncio
    async def test_no_cap_keeps_the_whole_string(self):
        message = await _wrap("a" * 50_000, max_result_chars=None)
        assert message.result.content == "a" * 50_000


class TestCallAndStream:
    @pytest.mark.asyncio
    async def test_streaming_tool_invalid_arguments_tagged_before_astream(
        self, registry
    ):
        tool = _StrictStreamingTool()
        tool.astream = AsyncMock()  # type: ignore[method-assign]
        registry.register(tool)
        call = ToolCall(id="c1", name="strict_streamer", arguments={})

        chunks = [c async for c in registry.call_and_stream(call)]

        assert len(chunks) == 1
        assert chunks[0].is_error is True
        assert chunks[0].is_validation_error is True
        assert "Invalid arguments" in chunks[0].delta
        tool.astream.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_streaming_tool_yields_single_final_chunk(self, registry):
        registry.register(_DummyTool())
        call = ToolCall(id="c1", name="dummy", arguments={})
        chunks = [c async for c in registry.call_and_stream(call)]
        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert chunks[0].is_error is False
        assert "handled" in chunks[0].delta

    @pytest.mark.asyncio
    async def test_streaming_tool_yields_deltas_then_final(self, registry):
        registry.register(_StreamingTool())
        call = ToolCall(id="c1", name="streamer", arguments={})
        chunks = [c async for c in registry.call_and_stream(call)]
        assert [c.delta for c in chunks] == ["hello ", "world", ""]
        assert chunks[-1].is_final is True
        assert chunks[-1].is_error is False

    @pytest.mark.asyncio
    async def test_streaming_tool_error_mid_stream(self, registry):
        registry.register(_FailingStreamingTool())
        call = ToolCall(id="c1", name="failing_streamer", arguments={})
        chunks = [c async for c in registry.call_and_stream(call)]
        assert chunks[0].delta == "partial"
        assert chunks[0].is_final is False
        assert chunks[-1].is_final is True
        assert chunks[-1].is_error is True
        assert "stream broke" in chunks[-1].delta

    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self, registry):
        call = ToolCall(id="c1", name="nonexistent", arguments={})
        with pytest.raises(ToolError, match="Unknown tool"):
            async for _ in registry.call_and_stream(call):
                pass

    @pytest.mark.asyncio
    async def test_streaming_tool_cancelled_error_propagates(self, registry):
        registry.register(_CancelledStreamingTool())
        call = ToolCall(id="c1", name="cancelled_streamer", arguments={})
        with pytest.raises(asyncio.CancelledError):
            async for _ in registry.call_and_stream(call):
                pass


class _SlowTool(Tool):
    name = "slow"
    description = "A tool that takes a while"
    input_schema = BaseModel

    def __init__(self, delay: float) -> None:
        self._delay = delay

    async def run(self, **kwargs):
        await asyncio.sleep(self._delay)
        return "done"


class _ConcurrencyTrackingTool(Tool):
    name = "tracked"
    description = "Tracks how many calls are in flight at once"
    input_schema = BaseModel

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def run(self, **kwargs):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return "done"


class TestPerToolTimeout:
    @pytest.mark.asyncio
    async def test_resolve_call_within_timeout_succeeds(self, registry):
        registry.register(_SlowTool(delay=0.001), timeout=1.0)
        call = ToolCall(id="c1", name="slow", arguments={})
        assert await registry.resolve_call(call) == "done"

    @pytest.mark.asyncio
    async def test_resolve_call_exceeding_timeout_raises(self, registry):
        registry.register(_SlowTool(delay=1.0), timeout=0.01)
        call = ToolCall(id="c1", name="slow", arguments={})
        with pytest.raises(ToolTimeoutError, match="timed out"):
            await registry.resolve_call(call)

    @pytest.mark.asyncio
    async def test_call_and_wrap_surfaces_timeout_as_error(self, registry):
        registry.register(_SlowTool(delay=1.0), timeout=0.01)
        call = ToolCall(id="c1", name="slow", arguments={})
        message = await registry.call_and_wrap(call)
        assert message.result.is_error is True
        assert is_tool_validation_error(message) is False

    @pytest.mark.asyncio
    async def test_no_timeout_means_unbounded(self, registry):
        registry.register(_SlowTool(delay=0.01))
        call = ToolCall(id="c1", name="slow", arguments={})
        assert await registry.resolve_call(call) == "done"

    def test_remove_clears_timeout(self, registry):
        registry.register(_SlowTool(delay=0.01), timeout=1.0)
        registry.remove("slow")
        registry.register(_SlowTool(delay=0.01))
        assert "slow" not in registry._timeouts


class TestPerToolConcurrency:
    @pytest.mark.asyncio
    async def test_max_concurrency_limits_in_flight_calls(self, registry):
        tool = _ConcurrencyTrackingTool()
        registry.register(tool, max_concurrency=2)
        calls = [ToolCall(id=str(i), name="tracked", arguments={}) for i in range(5)]

        await asyncio.gather(*(registry.call_and_wrap(c) for c in calls))

        assert tool.max_in_flight <= 2

    @pytest.mark.asyncio
    async def test_no_max_concurrency_means_unbounded(self, registry):
        tool = _ConcurrencyTrackingTool()
        registry.register(tool)
        calls = [ToolCall(id=str(i), name="tracked", arguments={}) for i in range(5)]

        await asyncio.gather(*(registry.call_and_wrap(c) for c in calls))

        assert tool.max_in_flight == 5

    @pytest.mark.asyncio
    async def test_max_concurrency_applies_to_streaming(self, registry):
        registry.register(_StreamingTool(), max_concurrency=1)
        call = ToolCall(id="c1", name="streamer", arguments={})
        chunks = [c async for c in registry.call_and_stream(call)]
        assert [c.delta for c in chunks] == ["hello ", "world", ""]

    def test_remove_clears_semaphore(self, registry):
        registry.register(_DummyTool(), max_concurrency=1)
        registry.remove("dummy")
        assert "dummy" not in registry._semaphores


class TestDiscoverEntryPoints:
    def test_discovers_and_registers_tools(self, registry, monkeypatch):
        class _FakeEntryPoint:
            def __init__(self, factory):
                self._factory = factory

            def load(self):
                return self._factory

        fake_entry_points = [_FakeEntryPoint(_DummyTool), _FakeEntryPoint(_OtherTool)]
        monkeypatch.setattr(
            "agent_platform.agents.tools.registry.entry_points",
            lambda group: fake_entry_points if group == "agent_platform.tools" else [],
        )

        discovered = registry.discover_entry_points()

        assert {t.name for t in discovered} == {"dummy", "other"}
        assert "dummy" in registry
        assert "other" in registry

    def test_uses_custom_group(self, registry, monkeypatch):
        calls = []

        def fake_entry_points(group):
            calls.append(group)
            return []

        monkeypatch.setattr(
            "agent_platform.agents.tools.registry.entry_points", fake_entry_points
        )

        registry.discover_entry_points(group="custom.group")

        assert calls == ["custom.group"]

    def test_no_entry_points_returns_empty(self, registry, monkeypatch):
        monkeypatch.setattr(
            "agent_platform.agents.tools.registry.entry_points", lambda group: []
        )
        assert registry.discover_entry_points() == []


class TestToolCallObservability:
    async def test_span_describes_the_tool_and_its_arguments(
        self, registry, recorded_spans
    ):
        registry.register(_StrictTool())

        await registry.call_and_wrap(
            ToolCall(id="c1", name="strict", arguments={"location": "Paris"})
        )

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "execute_tool"
        assert span.attributes["tool.name"] == "strict"
        assert span.attributes["tool.call.id"] == "c1"
        assert span.attributes["tool.kind"] == "function"
        assert span.attributes["tool.arguments.size"] > 0
        assert span.attributes["tool.arguments.digest"]
        # Off by default: arguments routinely carry user data.
        assert "tool.arguments.content" not in span.attributes

    async def test_span_describes_the_result(self, registry, recorded_spans):
        registry.register(_DummyTool())

        await registry.call_and_wrap(ToolCall(id="c1", name="dummy", arguments={}))

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["tool.result.is_error"] is False
        assert span.attributes["tool.result.size"] > 0

    async def test_a_failing_call_is_marked_with_its_error_type(
        self, registry, recorded_spans
    ):
        registry.register(_StrictTool())

        await registry.call_and_wrap(
            ToolCall(id="c1", name="strict", arguments={"wrong": 1})
        )

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["tool.result.is_error"] is True
        assert span.attributes["error.type"] == "ToolCallValidationError"

    async def test_an_unknown_tool_still_produces_a_span(
        self, registry, recorded_spans
    ):
        await registry.call_and_wrap(ToolCall(id="c1", name="nope", arguments={}))

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["tool.name"] == "nope"
        assert "tool.kind" not in span.attributes

    async def test_streaming_records_the_chunk_count_and_joined_result(
        self, registry, recorded_spans
    ):
        registry.register(_StreamingTool())

        async for _chunk in registry.call_and_stream(
            ToolCall(id="c1", name="streamer", arguments={})
        ):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["tool.stream.chunks"] == 2
        assert span.attributes["tool.result.size"] == len("hello world")
        assert span.attributes["tool.result.is_error"] is False

    async def test_a_failing_stream_is_marked_as_an_error(
        self, registry, recorded_spans
    ):
        registry.register(_FailingStreamingTool())

        async for _chunk in registry.call_and_stream(
            ToolCall(id="c1", name="failing_streamer", arguments={})
        ):
            pass

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["tool.result.is_error"] is True

    async def test_captured_content_lands_on_the_span_when_enabled(
        self, registry, recorded_spans, monkeypatch
    ):
        monkeypatch.setenv("AGENT_PLATFORM_CAPTURE_CONTENT", "1")
        registry.register(_StrictTool())

        await registry.call_and_wrap(
            ToolCall(id="c1", name="strict", arguments={"location": "Paris"})
        )

        (span,) = recorded_spans.get_finished_spans()
        assert "Paris" in span.attributes["tool.arguments.content"]
        assert span.attributes["tool.result.content"] == "weather for Paris"
