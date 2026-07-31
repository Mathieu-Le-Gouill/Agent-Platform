from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from agent_platform.core.interfaces.mcp.config import MCPClientConfig
from agent_platform.core.schemas.mcp import MCPToolSpec

ConfigT = TypeVar("ConfigT", bound=MCPClientConfig)


class BaseMCPClient(ABC, Generic[ConfigT]):
    """Talks to one MCP server: discovers its tools and invokes them.

    Unlike the other domains, an MCP server is a stateful session (a
    subprocess or a network stream), so `connect()`/`aclose()` are part of
    the contract rather than left to callers; `list_tools()`/`call_tool()`
    assume an already-connected client. The `async with client:` form is
    provided for free via `__aenter__`/`__aexit__`.
    """

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    @abstractmethod
    async def list_tools(self, config: ConfigT | None = None) -> list[MCPToolSpec]: ...

    @abstractmethod
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        config: ConfigT | None = None,
    ) -> Any: ...

    async def __aenter__(self) -> BaseMCPClient[ConfigT]:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
