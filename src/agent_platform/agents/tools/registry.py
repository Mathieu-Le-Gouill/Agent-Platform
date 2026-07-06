from __future__ import annotations

from typing import Any

from agent_platform.agents.tools.base import Tool, ToolError
from agent_platform.models.message import ToolCall


class ToolRegistry:
    _tools: dict[str, Tool]

    def __init__(self) -> None:
        self._tools = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool: '{name}'")
        return tool

    def all(self) -> dict[str, Tool]:
        return dict(self._tools)

    def remove(self, name: str) -> None:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: '{name}'")
        del self._tools[name]

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def anthropic_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_anthropic_schema() for tool in self._tools.values()]

    async def resolve_call(self, call: ToolCall) -> Any:
        tool = self.get(call.name)
        return await tool.run(**call.arguments)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
