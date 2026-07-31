from __future__ import annotations

from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.mcp.adapter import MCPToolAdapter
from agent_platform.core.interfaces.mcp.base import BaseMCPClient


async def discover_mcp_tools(client: BaseMCPClient) -> list[Tool]:
    """List every tool `client`'s MCP server exposes and wrap each as a `Tool`,
    ready for `ToolRegistry.register()`. `client` must already be connected
    (typically via `async with client:`, see `BaseMCPClient`).
    """
    specs = await client.list_tools()
    return [MCPToolAdapter(client, spec) for spec in specs]
