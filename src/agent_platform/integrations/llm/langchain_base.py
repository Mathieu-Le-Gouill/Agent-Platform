from __future__ import annotations

import base64
import json
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Generic

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage as LCBaseMessage,
    HumanMessage,
    SystemMessage as LCSystemMessage,
    AIMessage,
    ToolMessage as LCToolMessage,
)

from agent_platform.core.interfaces.llm.response import (
    LLMResponse,
    StreamChunk,
    FinishReason,
)
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.schemas.message import (
    AssistantMessage,
    AudioBlock,
    ContentBlock,
    ContentMessage,
    ImageBlock,
    SystemMessage,
    TextBlock,
    UserMessage,
    ToolMessage,
    ToolCall,
    Prompt,
)
from agent_platform.core.interfaces.llm.base import (
    BaseLLMProvider,
    CredentialsT,
    GenerationConfigT,
)
from agent_platform.core.tracing import TracingBackend, TracingConfig

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class LangChainLLMProvider(
    BaseLLMProvider[CredentialsT, GenerationConfigT],
    Generic[CredentialsT, GenerationConfigT],
):
    @abstractmethod
    def _client(self, config: GenerationConfigT) -> BaseChatModel: ...

    @abstractmethod
    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]: ...

    @abstractmethod
    def _default_config(self) -> GenerationConfigT: ...

    def generate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        lc = self._client(config)
        if tools:
            lc = lc.bind_tools([self._tool_to_schema(t) for t in tools])
        response = lc.invoke(
            _to_langchain(prompt), config={"callbacks": get_langchain_callbacks()}
        )
        return _from_langchain(response, config.model)

    async def agenerate(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
        tools: list[Tool] | None = None,
    ) -> LLMResponse:
        config = config or self._default_config()

        lc = self._client(config)
        if tools:
            lc = lc.bind_tools([self._tool_to_schema(t) for t in tools])
        response = await lc.ainvoke(
            _to_langchain(prompt), config={"callbacks": get_langchain_callbacks()}
        )
        return _from_langchain(response, config.model)

    async def stream(
        self,
        prompt: Prompt,
        config: GenerationConfigT | None = None,
    ) -> AsyncIterator[StreamChunk]:
        config = config or self._default_config()
        lc = self._client(config)

        async for chunk in lc.astream(
            _to_langchain(prompt), config={"callbacks": get_langchain_callbacks()}
        ):
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


def get_langchain_callbacks(config: TracingConfig | None = None) -> list[Any]:
    config = config or TracingConfig.from_env()
    match config.backend:
        case TracingBackend.LANGSMITH:
            # LangSmith traces automatically via LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY
            # env vars picked up internally by langchain-core; no explicit callback needed.
            return []
        case TracingBackend.LANGFUSE:
            from langfuse.callback import CallbackHandler

            # Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env.
            return [CallbackHandler()]
        case _:
            return []


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


def _content_to_langchain(m: ContentMessage) -> str | list[dict[str, Any]]:
    if isinstance(m.content, str):
        return m.content
    return [_block_to_langchain(b) for b in m.blocks]


def _to_langchain(prompt: Prompt) -> list[LCBaseMessage]:
    result = []
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
                    result.append(
                        AIMessage(content=content, tool_calls=lc_tool_calls)
                    )
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
