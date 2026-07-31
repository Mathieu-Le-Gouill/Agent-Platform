from typing import Any

import pytest

from agent_platform.agents.tools.mcp.adapter import MCPToolAdapter
from agent_platform.agents.tools.mcp.discovery import discover_mcp_tools
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.interfaces.mcp.base import BaseMCPClient
from agent_platform.core.schemas.mcp import MCPToolSpec


class _FakeMCPClient(BaseMCPClient):
    def __init__(self, specs: list[MCPToolSpec]) -> None:
        self._specs = specs

    async def connect(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def list_tools(self, config=None):
        return self._specs

    async def call_tool(self, name: str, arguments: dict[str, Any], config=None):
        return f"{name}-result"


class TestDiscoverMCPTools:
    @pytest.mark.asyncio
    async def test_returns_one_adapter_per_spec(self):
        specs = [
            MCPToolSpec(name="echo", input_schema={"type": "object"}),
            MCPToolSpec(name="add", input_schema={"type": "object"}),
        ]
        client = _FakeMCPClient(specs)

        tools = await discover_mcp_tools(client)

        assert len(tools) == 2
        assert all(isinstance(tool, MCPToolAdapter) for tool in tools)
        assert {tool.name for tool in tools} == {"echo", "add"}

    @pytest.mark.asyncio
    async def test_no_tools_returns_empty_list(self):
        client = _FakeMCPClient([])
        assert await discover_mcp_tools(client) == []

    @pytest.mark.asyncio
    async def test_discovered_tools_register_and_run(self):
        specs = [MCPToolSpec(name="echo", input_schema={"type": "object"})]
        client = _FakeMCPClient(specs)
        registry = ToolRegistry()

        for tool in await discover_mcp_tools(client):
            registry.register(tool)

        assert "echo" in registry
        assert await registry.get("echo").run() == "echo-result"
