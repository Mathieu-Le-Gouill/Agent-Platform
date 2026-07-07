from __future__ import annotations

import json
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage as LCBaseMessage,
    HumanMessage,
    SystemMessage as LCSystemMessage,
    AIMessage,
    ToolMessage as LCToolMessage,
)

from agent_platform.integrations.llm.config import GenerationConfig
from agent_platform.integrations.llm.response import (
    LLMResponse,
    StreamChunk,
    FinishReason,
)
from agent_platform.models.token import TokenUsage
from agent_platform.models.message import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
    ToolMessage,
    ToolCall,
    Prompt,
)
from agent_platform.integrations.llm.base import BaseLLMProvider

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class LangChainLLMProvider(BaseLLMProvider[GenerationConfig]):
    @abstractmethod
    def _client(self, model: str, config: GenerationConfig | None) -> BaseChatModel: ...

    @abstractmethod
    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]: ...

    async def generate(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        lc = self._client(model, config)
        if tools:
            lc = lc.bind_tools([self._tool_to_schema(t) for t in tools])
        response = await lc.ainvoke(_to_langchain(prompt))
        return _from_langchain(response, model)

    async def stream(
        self,
        prompt: Prompt,
        model: str,
        config: GenerationConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        lc = self._client(model, config)

        async for chunk in lc.astream(_to_langchain(prompt)):
            content = chunk.content
            if isinstance(content, str):
                if content:
                    yield StreamChunk(delta=content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str) and item:
                        yield StreamChunk(delta=item)
                    elif isinstance(item, dict):
                        text = item.get("text", "")
                        if isinstance(text, str) and text:
                            yield StreamChunk(delta=text)

            usage = chunk.usage_metadata
            if usage:
                yield StreamChunk(
                    delta="",
                    finish_reason=FinishReason.STOP,
                    usage=TokenUsage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    ),
                )

        yield StreamChunk(delta="", finish_reason=FinishReason.STOP)


# --- Mappers ---


def _to_langchain(prompt: Prompt) -> list[LCBaseMessage]:
    result = []
    for m in prompt.messages:
        match m:
            case SystemMessage():
                result.append(LCSystemMessage(content=m.content))
            case UserMessage():
                result.append(HumanMessage(content=m.content))
            case AssistantMessage():
                if m.tool_calls:
                    lc_tool_calls: list[dict[str, Any]] = [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                    result.append(
                        AIMessage(content=m.content, tool_calls=lc_tool_calls)
                    )
                else:
                    result.append(AIMessage(content=m.content))
            case ToolMessage():
                result.append(
                    LCToolMessage(
                        content=m.result.content,
                        tool_call_id=m.result.tool_call_id,
                    )
                )
    return result


def _from_langchain(response: AIMessage, model: str) -> LLMResponse:
    tool_calls = [
        ToolCall(
            id=tc["id"] or "",
            name=tc["name"],
            arguments=tc["args"],
        )
        for tc in (response.tool_calls or [])
    ]

    usage_meta = response.usage_metadata or {}

    if isinstance(response.content, str):
        content = response.content
    else:
        content = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in response.content
        )

    return LLMResponse(
        message=AssistantMessage(content=content, tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=usage_meta.get("input_tokens", 0),
            output_tokens=usage_meta.get("output_tokens", 0),
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )
