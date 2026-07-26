from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from opentelemetry.trace import Span, Status, StatusCode

from agent_platform.agents.tools.base import Tool, ToolStreamChunk
from agent_platform.agents.tools.errors import ToolNotFoundError, ToolRegistrationError
from agent_platform.core.schemas.message import (
    ToolCall,
    ToolMessage,
    ToolResult,
)
from agent_platform.core.tracing import GenAIAttributes, traced_operation_span

logger = logging.getLogger(__name__)


class ToolRegistry:
    _tools: dict[str, Tool]

    def __init__(self) -> None:
        self._tools = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolRegistrationError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Unknown tool: '{name}'")
        return tool

    def all(self) -> dict[str, Tool]:
        return dict(self._tools)

    def remove(self, name: str) -> None:
        if name not in self._tools:
            raise ToolNotFoundError(f"Unknown tool: '{name}'")
        del self._tools[name]

    async def resolve_call(self, call: ToolCall) -> Any:
        tool = self.get(call.name)
        return await tool.run(**call.arguments)

    @staticmethod
    def _record_tool_error(span: Span, exc: Exception) -> None:
        """Mark `span` as failed the way the GenAI conventions expect.

        Tool errors are caught here rather than left to propagate, so
        `traced_span`'s own except-block (which does this automatically for
        propagating exceptions) never runs; set `error.type` and the span
        status by hand instead.
        """
        span.set_attribute(GenAIAttributes.ERROR_TYPE, type(exc).__qualname__)
        span.set_status(Status(StatusCode.ERROR, str(exc)))

    async def call_and_wrap(self, call: ToolCall) -> ToolMessage:
        with traced_operation_span(
            "execute_tool",
            **{
                GenAIAttributes.TOOL_NAME: call.name,
                GenAIAttributes.TOOL_CALL_ID: call.id,
            },
        ) as span:
            try:
                raw = await self.resolve_call(call)
                content = str(raw) if raw is not None else ""
                is_error = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Tool call '%s' failed", call.name)
                content = str(exc)
                is_error = True
                self._record_tool_error(span, exc)

        return ToolMessage(
            result=ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=content,
                is_error=is_error,
            )
        )

    async def call_and_stream(self, call: ToolCall) -> AsyncIterator[ToolStreamChunk]:
        tool = self.get(call.name)

        if not tool.supports_streaming:
            message = await self.call_and_wrap(call)
            yield ToolStreamChunk(
                tool_call_id=call.id,
                delta=message.result.content,
                is_final=True,
                is_error=message.result.is_error,
            )
            return

        with traced_operation_span(
            "execute_tool",
            **{
                GenAIAttributes.TOOL_NAME: call.name,
                GenAIAttributes.TOOL_CALL_ID: call.id,
            },
        ) as span:
            try:
                async for delta in tool.astream(**call.arguments):
                    yield ToolStreamChunk(tool_call_id=call.id, delta=delta)
                yield ToolStreamChunk(tool_call_id=call.id, delta="", is_final=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Tool stream '%s' failed", call.name)
                self._record_tool_error(span, exc)
                yield ToolStreamChunk(
                    tool_call_id=call.id, delta=str(exc), is_final=True, is_error=True
                )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
