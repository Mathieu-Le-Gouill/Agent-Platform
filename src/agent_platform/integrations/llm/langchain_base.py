from __future__ import annotations

import base64
import json
from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Generic

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    UsageMetadata,
)
from langchain_core.messages import (
    BaseMessage as LCBaseMessage,
)
from langchain_core.messages import (
    SystemMessage as LCSystemMessage,
)
from langchain_core.messages import (
    ToolMessage as LCToolMessage,
)
from langchain_core.runnables import Runnable

from agent_platform.core.errors import ProviderError, error_logged, with_retry
from agent_platform.core.interfaces.llm.base import (
    BaseLLMProvider,
    GenerationConfigT,
)
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    StreamChunk,
)
from agent_platform.core.schemas.message import (
    AssistantMessage,
    AudioBlock,
    ContentBlock,
    ContentMessage,
    ImageBlock,
    Prompt,
    SystemMessage,
    TextBlock,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.tracing import (
    GenAIAttributes,
    record_token_usage,
    traced_operation_span,
)

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class LangChainLLMProvider(
    BaseLLMProvider[GenerationConfigT],
    Generic[GenerationConfigT],
):
    @abstractmethod
    def _client(self, config: GenerationConfigT) -> BaseChatModel: ...

    @abstractmethod
    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]: ...

    @abstractmethod
    def _default_config(self) -> GenerationConfigT: ...

    def _gen_ai_system(self) -> str:
        name = type(self).__name__
        return (name[: -len("LLM")] if name.endswith("LLM") else name).lower()

    def generate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        with traced_operation_span(
            "chat",
            **{
                GenAIAttributes.SYSTEM: self._gen_ai_system(),
                GenAIAttributes.REQUEST_MODEL: config.model,
            },
        ) as span:
            client = self._client(config)
            runnable: Runnable[Any, Any] = (
                client.bind_tools([self._tool_to_schema(t) for t in tools])
                if tools
                else client
            )
            response = runnable.invoke(_to_langchain(prompt))
            result = _from_langchain(response, config.model)
            record_token_usage(span, result.usage)
            return result

    @error_logged(re_raise=ProviderError, message="LLM generation failed")
    @with_retry()
    async def agenerate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        with traced_operation_span(
            "chat",
            **{
                GenAIAttributes.SYSTEM: self._gen_ai_system(),
                GenAIAttributes.REQUEST_MODEL: config.model,
            },
        ) as span:
            client = self._client(config)
            runnable: Runnable[Any, Any] = (
                client.bind_tools([self._tool_to_schema(t) for t in tools])
                if tools
                else client
            )
            response = await runnable.ainvoke(_to_langchain(prompt))
            result = _from_langchain(response, config.model)
            record_token_usage(span, result.usage)
            return result

    async def stream(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        lc = self._client(config)

        with traced_operation_span(
            "chat",
            **{
                GenAIAttributes.SYSTEM: self._gen_ai_system(),
                GenAIAttributes.REQUEST_MODEL: config.model,
            },
        ) as span:
            usage_totals = TokenUsage.zero()
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
                    usage_totals = usage_totals + TokenUsage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    )
                    yield StreamChunk(
                        delta="",
                        finish_reason=FinishReason.STOP,
                        usage=TokenUsage(
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                        ),
                    )

            record_token_usage(span, usage_totals)
            yield StreamChunk(delta="", finish_reason=FinishReason.STOP)


# --- Mappers ---


def _block_to_langchain(block: ContentBlock) -> dict[str, Any]:
    match block:
        case TextBlock():
            return {"type": "text", "text": block.text}
        case ImageBlock():
            if isinstance(block.image, str):
                url = block.image
            else:
                mime = (
                    f"image/{block.image.format.value}"
                    if block.image.format
                    else "image/png"
                )
                data = base64.b64encode(block.image.content).decode()
                url = f"data:{mime};base64,{data}"
            return {"type": "image_url", "image_url": {"url": url}}
        case AudioBlock():
            data = base64.b64encode(block.audio.content).decode()
            fmt = block.audio.format.value if block.audio.format else "wav"
            return {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}


def _content_to_langchain(m: ContentMessage) -> str | list[str | dict[str, Any]]:
    if isinstance(m.content, str):
        return m.content
    return [_block_to_langchain(b) for b in m.blocks]


def _to_langchain(prompt: Prompt) -> list[LCBaseMessage]:
    result: list[LCBaseMessage] = []
    for m in prompt.messages:
        match m:
            case SystemMessage():
                result.append(LCSystemMessage(content=_content_to_langchain(m)))
            case UserMessage():
                result.append(HumanMessage(content=_content_to_langchain(m)))
            case AssistantMessage():
                content = _content_to_langchain(m)
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
                    result.append(AIMessage(content=content, tool_calls=lc_tool_calls))
                else:
                    result.append(AIMessage(content=content))
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

    usage_meta: UsageMetadata | dict[str, Any] = response.usage_metadata or {}

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
