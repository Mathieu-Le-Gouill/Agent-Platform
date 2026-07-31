from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from importlib.metadata import entry_points
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from agent_platform.agents.tools.base import Tool, ToolStreamChunk
from agent_platform.agents.tools.errors import (
    ToolCallValidationError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolTimeoutError,
)
from agent_platform.core.genai_tracing import GenAIAttributes, traced_operation_span
from agent_platform.core.schemas.message import (
    ToolCall,
    ToolMessage,
    ToolResult,
)
from agent_platform.core.tracing import mark_span_error

logger = logging.getLogger(__name__)

TOOL_CALL_VALIDATION_ERROR_KEY = "tool_call_validation_error"


def is_tool_validation_error(message: ToolMessage) -> bool:
    return (
        message.result.is_error
        and message.metadata.get(TOOL_CALL_VALIDATION_ERROR_KEY) is True
    )


class ToolRegistry:
    _tools: dict[str, Tool]

    def __init__(self) -> None:
        self._tools = {}
        self._timeouts: dict[str, float] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    def register(
        self,
        tool: Tool,
        *,
        timeout: float | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        """Register `tool`, optionally guarding it with a per-call `timeout`
        (seconds, enforced on `run()` only, see `resolve_call`) and/or a
        `max_concurrency` cap on simultaneous in-flight calls (enforced on both
        `run()` and `astream()`), both opt-in and independent of one another.
        """
        if tool.name in self._tools:
            raise ToolRegistrationError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        if timeout is not None:
            self._timeouts[tool.name] = timeout
        if max_concurrency is not None:
            self._semaphores[tool.name] = asyncio.Semaphore(max_concurrency)

    def discover_entry_points(
        self, *, group: str = "agent_platform.tools"
    ) -> list[Tool]:
        """Register every `Tool` exposed by an installed package as an
        `importlib.metadata` entry point under `group`. Each entry point must
        resolve to a zero-argument callable (a class implementing `Tool`, or a
        factory function) returning a `Tool` instance, so third-party packages
        can ship tools discoverable at startup without this registry knowing
        about them ahead of time. Returns the tools registered, in entry-point
        iteration order.
        """
        discovered: list[Tool] = []
        for entry_point in entry_points(group=group):
            factory = entry_point.load()
            tool = factory()
            self.register(tool)
            discovered.append(tool)
        return discovered

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
        self._timeouts.pop(name, None)
        self._semaphores.pop(name, None)

    async def resolve_call(self, call: ToolCall) -> Any:
        tool = self.get(call.name)
        self._validate_arguments(tool, call)
        timeout = self._timeouts.get(call.name)
        if timeout is None:
            return await tool.run(**call.arguments)
        try:
            return await asyncio.wait_for(tool.run(**call.arguments), timeout=timeout)
        except TimeoutError as exc:
            raise ToolTimeoutError(
                f"Tool '{call.name}' timed out after {timeout}s"
            ) from exc

    def _validate_arguments(self, tool: Tool, call: ToolCall) -> None:
        if tool.input_schema is BaseModel:
            return
        try:
            tool.input_schema(**call.arguments)
        except PydanticValidationError as exc:
            raise ToolCallValidationError(
                f"Invalid arguments for tool '{call.name}': {exc}"
            ) from exc

    def _concurrency_guard(self, name: str) -> contextlib.AbstractAsyncContextManager:
        semaphore = self._semaphores.get(name)
        return semaphore if semaphore is not None else contextlib.nullcontext()

    async def call_and_wrap(self, call: ToolCall) -> ToolMessage:
        async with self._concurrency_guard(call.name):
            with traced_operation_span(
                "execute_tool",
                {
                    GenAIAttributes.TOOL_NAME: call.name,
                    GenAIAttributes.TOOL_CALL_ID: call.id,
                },
            ) as span:
                metadata: dict[str, Any] = {}
                try:
                    raw = await self.resolve_call(call)
                    content = str(raw) if raw is not None else ""
                    is_error = False
                except asyncio.CancelledError:
                    raise
                except ToolCallValidationError as exc:
                    logger.info("Tool call '%s' failed argument validation", call.name)
                    content = str(exc)
                    is_error = True
                    metadata = {TOOL_CALL_VALIDATION_ERROR_KEY: True}
                    mark_span_error(span, exc, record_exception=False)
                except Exception as exc:
                    logger.exception("Tool call '%s' failed", call.name)
                    content = str(exc)
                    is_error = True
                    mark_span_error(span, exc, record_exception=False)

        return ToolMessage(
            result=ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=content,
                is_error=is_error,
            ),
            metadata=metadata,
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
                is_validation_error=is_tool_validation_error(message),
            )
            return

        async with self._concurrency_guard(call.name):
            with traced_operation_span(
                "execute_tool",
                {
                    GenAIAttributes.TOOL_NAME: call.name,
                    GenAIAttributes.TOOL_CALL_ID: call.id,
                },
            ) as span:
                try:
                    self._validate_arguments(tool, call)
                    async for delta in tool.astream(**call.arguments):
                        yield ToolStreamChunk(tool_call_id=call.id, delta=delta)
                    yield ToolStreamChunk(tool_call_id=call.id, delta="", is_final=True)
                except asyncio.CancelledError:
                    raise
                except ToolCallValidationError as exc:
                    logger.info("Tool call '%s' failed argument validation", call.name)
                    mark_span_error(span, exc, record_exception=False)
                    yield ToolStreamChunk(
                        tool_call_id=call.id,
                        delta=str(exc),
                        is_final=True,
                        is_error=True,
                        is_validation_error=True,
                    )
                except Exception as exc:
                    logger.exception("Tool stream '%s' failed", call.name)
                    mark_span_error(span, exc, record_exception=False)
                    yield ToolStreamChunk(
                        tool_call_id=call.id,
                        delta=str(exc),
                        is_final=True,
                        is_error=True,
                    )

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
