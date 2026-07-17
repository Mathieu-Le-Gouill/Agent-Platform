from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from agent_platform.agents.tools import Tool, ToolError, ToolRegistry
from agent_platform.core.schemas.message import ToolCall


class _DummyTool(Tool):
    name = "dummy"
    description = "A dummy tool"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return {"handled": True}


class _OtherTool(Tool):
    name = "other"
    description = "Another tool"
    input_schema = BaseModel

    async def run(self, **kwargs):
        return None


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
