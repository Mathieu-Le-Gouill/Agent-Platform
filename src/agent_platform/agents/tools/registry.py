from __future__ import annotations

from typing import Any

from agent_platform.agents.tools.base import Tool, ToolError
from agent_platform.models.message import (
    ToolCall,
    ToolMessage,
    ToolResult,
)


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

    async def resolve_call(self, call: ToolCall) -> Any:
        tool = self.get(call.name)
        return await tool.run(**call.arguments)

    async def call_and_wrap(self, call: ToolCall) -> ToolMessage:
        try:
            raw = await self.resolve_call(call)
            content = str(raw) if raw is not None else ""
            is_error = False
        except Exception as exc:
            content = str(exc)
            is_error = True

        return ToolMessage(
            result=ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=content,
                is_error=is_error,
            )
        )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
