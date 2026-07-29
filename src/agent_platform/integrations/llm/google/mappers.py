from __future__ import annotations

from typing import Any
from uuid import uuid4

from google.genai import types

from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    ResponseFormat,
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
from agent_platform.integrations.llm.google.config import GoogleGenerationConfig

__all__ = [
    "block_to_native_part",
    "content_to_native_parts",
    "from_native_response",
    "to_native_config",
    "to_native_contents",
]


def block_to_native_part(block: ContentBlock) -> types.Part:
    match block:
        case TextBlock():
            return types.Part.from_text(text=block.text)
        case ImageBlock():
            if isinstance(block.image, str):
                return types.Part.from_uri(file_uri=block.image, mime_type="image/png")
            mime = (
                f"image/{block.image.format.value}"
                if block.image.format
                else "image/png"
            )
            return types.Part.from_bytes(data=block.image.content, mime_type=mime)
        case AudioBlock():
            mime = block.audio.format.value if block.audio.format else "wav"
            return types.Part.from_bytes(
                data=block.audio.content, mime_type=f"audio/{mime}"
            )


def content_to_native_parts(m: ContentMessage) -> list[types.Part]:
    if isinstance(m.content, str):
        return [types.Part.from_text(text=m.content)] if m.content else []
    return [block_to_native_part(b) for b in m.blocks]


def to_native_contents(prompt: Prompt) -> tuple[str | None, list[types.Content]]:
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for m in prompt.messages:
        match m:
            case SystemMessage():
                if m.text:
                    system_parts.append(m.text)
            case UserMessage():
                contents.append(
                    types.Content(role="user", parts=content_to_native_parts(m))
                )
            case AssistantMessage():
                parts = content_to_native_parts(m)
                for tc in m.tool_calls:
                    parts.append(
                        types.Part.from_function_call(name=tc.name, args=tc.arguments)
                    )
                contents.append(types.Content(role="model", parts=parts))
            case ToolMessage():
                # Gemini folds tool results into a user-role turn, matched by
                # function name rather than a call id (it has none).
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=m.result.name,
                                response={"content": m.result.content},
                            )
                        ],
                    )
                )

    return ("\n\n".join(system_parts) if system_parts else None, contents)


def to_native_config(
    config: GoogleGenerationConfig,
    system: str | None,
    tools: list[Any] | None,
    tool_to_schema: Any,
) -> types.GenerateContentConfig:
    kwargs: dict[str, Any] = {}
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.max_tokens is not None:
        kwargs["max_output_tokens"] = config.max_tokens
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        kwargs["top_k"] = config.top_k
    if config.stop_sequences:
        kwargs["stop_sequences"] = config.stop_sequences
    if config.seed is not None:
        kwargs["seed"] = config.seed
    if config.presence_penalty is not None:
        kwargs["presence_penalty"] = config.presence_penalty
    if config.frequency_penalty is not None:
        kwargs["frequency_penalty"] = config.frequency_penalty
    if system:
        kwargs["system_instruction"] = system
    if tools:
        kwargs["tools"] = [tool_to_schema(t) for t in tools]
    if config.thinking_budget is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=config.thinking_budget,
            include_thoughts=config.include_thoughts,
        )

    match config.response_format:
        case ResponseFormat.TEXT:
            pass
        case ResponseFormat.JSON:
            kwargs["response_mime_type"] = "application/json"
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = config.json_schema

    kwargs.update(config.extra_params)
    return types.GenerateContentConfig(**kwargs)


def from_native_response(
    response: types.GenerateContentResponse, model: str
) -> LLMResponse:
    candidate = response.candidates[0] if response.candidates else None
    content = candidate.content if candidate else None
    parts = content.parts if content else []

    text_parts = [part.text for part in (parts or []) if part.text]
    tool_calls = [
        ToolCall(
            id=part.function_call.id or uuid4().hex,
            name=part.function_call.name or "",
            arguments=dict(part.function_call.args or {}),
        )
        for part in (parts or [])
        if part.function_call
    ]

    usage = response.usage_metadata

    return LLMResponse(
        message=AssistantMessage(content="".join(text_parts), tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )
