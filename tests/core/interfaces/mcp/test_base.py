import pytest

from agent_platform.core.interfaces.mcp.base import BaseMCPClient
from agent_platform.core.schemas.mcp import MCPToolSpec


class _RecordingMCPClient(BaseMCPClient):
    def __init__(self) -> None:
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def aclose(self) -> None:
        self.closed = True

    async def list_tools(self, config=None):
        return [MCPToolSpec(name="echo", input_schema={"type": "object"})]

    async def call_tool(self, name, arguments, config=None):
        return f"{name}:{arguments}"


class TestBaseMCPClient:
    @pytest.mark.asyncio
    async def test_async_context_manager_connects_and_closes(self):
        client = _RecordingMCPClient()
        async with client as entered:
            assert entered is client
            assert client.connected is True
            assert client.closed is False
        assert client.closed is True

    @pytest.mark.asyncio
    async def test_list_tools_and_call_tool(self):
        client = _RecordingMCPClient()
        tools = await client.list_tools()
        assert tools == [MCPToolSpec(name="echo", input_schema={"type": "object"})]
        result = await client.call_tool("echo", {"x": 1})
        assert result == "echo:{'x': 1}"
