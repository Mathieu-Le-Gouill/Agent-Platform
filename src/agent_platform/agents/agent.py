from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from agent_platform.agents.tools.base import ToolStreamChunk
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.agents.errors import AgentThinkError
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.schemas.message import (
    AssistantMessage,
    Message,
    Prompt,
    ToolMessage,
    ToolResult,
)

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        *,
        name: str,
        llm: BaseLLMProvider,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        model: str = "default",
        generation_config: GenerationConfig | None = None,
    ) -> None:
        if not name:
            raise ValueError("Agent name must not be empty")
        self._name = name
        self._llm = llm
        self._tool_registry = tool_registry or ToolRegistry()
        self._system_prompt = system_prompt
        self._model = model
        self._generation_config = generation_config

    @property
    def name(self) -> str:
        return self._name

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    def _build_prompt(self, messages: list[Message]) -> Prompt:
        return Prompt.build(system=self._system_prompt, history=messages)

    async def think(self, messages: list[Message]) -> AssistantMessage:
        tool_list = list(self._tool_registry)

        try:
            response = await self._llm.agenerate(
                prompt=self._build_prompt(messages),
                model=self._model,
                config=self._generation_config,
                tools=tool_list or None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("LLM generation failed in agent '%s'", self._name)
            raise AgentThinkError(f"LLM generation failed: {exc}") from exc

        if response.message is None:
            raise AgentThinkError("LLM returned empty response")

        return response.message

    async def act(self, assistant_message: AssistantMessage) -> list[ToolMessage]:
        return [
            await self._tool_registry.call_and_wrap(tc)
            for tc in assistant_message.tool_calls
        ]

    async def act_stream(
        self, assistant_message: AssistantMessage
    ) -> AsyncIterator[ToolStreamChunk | ToolMessage]:
        for tc in assistant_message.tool_calls:
            content = ""
            is_error = False
            async for chunk in self._tool_registry.call_and_stream(tc):
                yield chunk
                if chunk.is_final:
                    is_error = chunk.is_error
                    if is_error:
                        content = chunk.delta
                elif not chunk.is_error:
                    content += chunk.delta

            yield ToolMessage(
                result=ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=content,
                    is_error=is_error,
                )
            )

    async def step(
        self, messages: list[Message]
    ) -> tuple[AssistantMessage, list[ToolMessage]]:
        assistant_msg = await self.think(messages)
        if not assistant_msg.tool_calls:
            return assistant_msg, []
        tool_messages = await self.act(assistant_msg)
        return assistant_msg, tool_messages
