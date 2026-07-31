import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from agent_platform.agents.tools import (
    Tool,
    ToolCallValidationError,
    ToolError,
    ToolRegistry,
    is_tool_validation_error,
)
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


class TestCallAndStream:
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
