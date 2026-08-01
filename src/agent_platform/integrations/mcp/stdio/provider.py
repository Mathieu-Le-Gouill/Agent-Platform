from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.mcp.base import BaseMCPClient
from agent_platform.core.interfaces.mcp.config import MCPClientConfig
from agent_platform.core.schemas.mcp import MCPToolSpec


class StdioMCPClient(BaseMCPClient[MCPClientConfig]):
    """Connects to an MCP server launched as a local subprocess over stdio,
    the transport the official `mcp` SDK's `stdio_client` implements.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._server_params = StdioServerParameters(
            command=command, args=args or [], env=env
        )
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        exit_stack = AsyncExitStack()
        read, write = await exit_stack.enter_async_context(
            stdio_client(self._server_params)
        )
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._exit_stack = exit_stack
        self._session = session

    async def aclose(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise ProviderError(
                "StdioMCPClient is not connected; call connect() or use 'async with'"
            )
        return self._session

    async def list_tools(
        self, config: MCPClientConfig | None = None
    ) -> list[MCPToolSpec]:
        session = self._require_session()
        result = await session.list_tools()
        return [
            MCPToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in result.tools
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        config: MCPClientConfig | None = None,
    ) -> Any:
        session = self._require_session()
        result = await session.call_tool(name, arguments)
        # Content blocks can be text, image, or resource types; only text ones
        # contribute to the joined result.
        text = "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        )
        if result.isError:
            raise ProviderError(f"MCP tool '{name}' returned an error: {text}")
        return text
