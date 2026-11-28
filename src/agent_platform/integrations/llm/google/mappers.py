from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any
from uuid import uuid4

from google.genai import types

from agent_platform.core.interfaces.llm.response import (
    LLMResponse,
    ResponseFormat,
)
from agent_platform.core.interfaces.llm.tool_spec import ToolSpec
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
from agent_platform.integrations.llm._finish_reason import to_finish_reason
from agent_platform.integrations.llm._generic_fields import (
    levels_for_model,
    nearest_level,
    warn_unsupported,
)
from agent_platform.integrations.llm._media import audio_format_name, image_mime_type
from agent_platform.integrations.llm.google.config import GoogleGenerationConfig

__all__ = [
    "block_to_native_part",
    "content_to_native_parts",
    "from_native_response",
    "to_native_config",
    "to_native_contents",
]

logger = logging.getLogger(__name__)

# `thinking_level` values each model accepts, by longest model-id prefix.
# https://aistudio.google.com/docs/thinking (model table). The docs also list
# `xhigh` for gemini-4-neon and gemini-4-hydrogen, but `types.ThinkingLevel`
# in the installed google-genai has no such value, so those map `xhigh` to
# `high`. A model matching no prefix gets the four levels the SDK can send.
_LOW_TO_HIGH = frozenset({"low", "medium", "high"})
_MINIMAL_TO_HIGH = frozenset({"minimal", "low", "medium", "high"})
_THINKING_LEVELS: dict[str, frozenset[str]] = {
    "gemini-4": _LOW_TO_HIGH,
    "gemini-3.8-flash": _LOW_TO_HIGH,
    "gemini-3.6-flash": _MINIMAL_TO_HIGH,
    "gemini-3.5-flash-lite": _MINIMAL_TO_HIGH,
    "gemini-3.1-flash-lite": _MINIMAL_TO_HIGH,
    "gemini-3.1-flash-lite-image": frozenset({"minimal", "high"}),
    "gemini-3.1-pro": _LOW_TO_HIGH,
    "gemini-3-flash": _MINIMAL_TO_HIGH,
    "gemini-3-pro": frozenset({"low", "high"}),
    "gemini-2.5": _LOW_TO_HIGH,
}


def block_to_native_part(block: ContentBlock) -> types.Part:
    match block:
        case TextBlock():
            return types.Part.from_text(text=block.text)
        case ImageBlock():
            if isinstance(block.image, str):
                return types.Part.from_uri(file_uri=block.image, mime_type="image/png")
            return types.Part.from_bytes(
                data=block.image.content, mime_type=image_mime_type(block.image)
            )
        case AudioBlock():
            return types.Part.from_bytes(
                data=block.audio.content,
                mime_type=f"audio/{audio_format_name(block.audio)}",
            )


def content_to_native_parts(m: ContentMessage) -> list[types.Part]:
    if isinstance(m.content, str):
        return [types.Part.from_text(text=m.content)] if m.content else []
    return [block_to_native_part(b) for b in m.blocks]


def to_native_contents(
    prompt: Prompt,
) -> tuple[str | None, list[types.ContentUnionDict]]:
    system_parts: list[str] = []
    # Typed as the SDK's element type, not `Content`: `list` is invariant, so
    # `generate_content(contents=...)` rejects a `list[Content]`.
    contents: list[types.ContentUnionDict] = []

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


def _thinking_config(config: GoogleGenerationConfig) -> types.ThinkingConfig | None:
    """Returns the thinking config: the provider budget, else the generic level.

    The generic level is clamped to the `thinking_level` values the model
    accepts.
    """
    if config.thinking_budget is not None:
        return types.ThinkingConfig(
            thinking_budget=config.thinking_budget,
            include_thoughts=config.include_thoughts,
        )
    level = config.reasoning_effort
    if level is None:
        return None
    supported = levels_for_model(config.model, _THINKING_LEVELS) or _MINIMAL_TO_HIGH
    native_level = types.ThinkingLevel(
        nearest_level("google", level, supported, config.model).upper()
    )
    return types.ThinkingConfig(
        thinking_level=native_level, include_thoughts=config.include_thoughts
    )


def to_native_config(
    config: GoogleGenerationConfig,
    system: str | None,
    tools: Sequence[ToolSpec] | None,
    tool_to_schema: Callable[[ToolSpec], types.Tool],
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
        if config.parallel_tool_calls is not None:
            warn_unsupported("google", "parallel_tool_calls")
    thinking = _thinking_config(config)
    if thinking is not None:
        kwargs["thinking_config"] = thinking

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

    # A thought part (`include_thoughts`) is the model's reasoning summary, not the answer.
    text_parts = [part.text for part in (parts or []) if part.text and not part.thought]
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
        finish_reason=to_finish_reason(candidate.finish_reason if candidate else None),
        response_id=response.response_id,
    )
