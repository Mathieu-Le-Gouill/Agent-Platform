from __future__ import annotations

import base64
from typing import Any

from anthropic.types import Message

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.llm.batch import BatchJob, BatchStatus
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
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig

__all__ = [
    "batch_job_from_native",
    "block_to_native",
    "content_to_native",
    "from_native_response",
    "to_native_messages",
    "to_native_params",
]

_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_THINKING_BUDGET = 5000
# Higher fallback so the default thinking_budget stays comfortably below
# max_tokens (Anthropic requires budget_tokens < max_tokens or the request 400s).
_DEFAULT_MAX_TOKENS_WITH_THINKING = 8192

_BATCH_STATUS_MAP: dict[str, BatchStatus] = {
    "in_progress": BatchStatus.IN_PROGRESS,
    "canceling": BatchStatus.IN_PROGRESS,
    "ended": BatchStatus.COMPLETED,
}


def batch_job_from_native(batch: Any) -> BatchJob:
    counts = getattr(batch, "request_counts", None)
    completed = None
    if counts is not None:
        completed = counts.succeeded + counts.errored + counts.canceled + counts.expired
    total = None
    if counts is not None:
        total = completed + counts.processing if completed is not None else None
    return BatchJob(
        id=batch.id,
        status=_BATCH_STATUS_MAP.get(batch.processing_status, BatchStatus.PENDING),
        request_count=total,
        completed_count=completed,
    )


def block_to_native(block: ContentBlock) -> dict[str, Any]:
    match block:
        case TextBlock():
            return {"type": "text", "text": block.text}
        case ImageBlock():
            if isinstance(block.image, str):
                return {"type": "image", "source": {"type": "url", "url": block.image}}
            mime = (
                f"image/{block.image.format.value}"
                if block.image.format
                else "image/png"
            )
            data = base64.b64encode(block.image.content).decode()
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": data},
            }
        case AudioBlock():
            raise ProviderError("Anthropic does not support audio content blocks")


def content_to_native(m: ContentMessage) -> str | list[dict[str, Any]]:
    if isinstance(m.content, str):
        return m.content
    return [block_to_native(b) for b in m.blocks]


def to_native_messages(prompt: Prompt) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []

    for m in prompt.messages:
        match m:
            case SystemMessage():
                if m.text:
                    system_parts.append(m.text)
            case UserMessage():
                messages.append({"role": "user", "content": content_to_native(m)})
            case AssistantMessage():
                content = [block_to_native(b) for b in m.blocks]
                for tc in m.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                messages.append({"role": "assistant", "content": content})
            case ToolMessage():
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.result.tool_call_id,
                                "content": m.result.content,
                                "is_error": m.result.is_error,
                            }
                        ],
                    }
                )

    return ("\n\n".join(system_parts) if system_parts else None, messages)


def to_native_params(config: AnthropicGenerationConfig) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if config.temperature is not None:
        params["temperature"] = config.temperature
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.top_k is not None:
        params["top_k"] = config.top_k
    if config.stop_sequences:
        params["stop_sequences"] = config.stop_sequences

    output_config: dict[str, Any] = {}

    if config.effort is not None:
        # Modern effort control supersedes manual thinking_budget management;
        # skip the legacy `thinking` payload entirely when set.
        params["max_tokens"] = config.max_tokens or _DEFAULT_MAX_TOKENS
        output_config["effort"] = config.effort
    elif config.thinking:
        budget_tokens = config.thinking_budget or _DEFAULT_THINKING_BUDGET
        if config.max_tokens is not None:
            max_tokens = config.max_tokens
            if budget_tokens >= max_tokens:
                budget_tokens = max(1, max_tokens - 1024)
        else:
            # No explicit max_tokens: size it to comfortably fit the (possibly
            # user-supplied) budget rather than clamping the user's budget down
            # to an unrelated fixed default.
            max_tokens = max(_DEFAULT_MAX_TOKENS_WITH_THINKING, budget_tokens + 1024)
        params["max_tokens"] = max_tokens
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }
    else:
        params["max_tokens"] = config.max_tokens or _DEFAULT_MAX_TOKENS

    if config.cache_control:
        # Native top-level cache_control param on messages.create, no more
        # model_kwargs pass-through hack required by LangChain's ChatAnthropic.
        params["cache_control"] = {"type": "ephemeral"}

    match config.response_format:
        case ResponseFormat.TEXT:
            pass
        case ResponseFormat.JSON:
            # Anthropic's native structured-output mode (`output_config.format`)
            # only has a `json_schema` variant, no schema-less "json_object"
            # mode like OpenAI/Mistral; raise instead of silently ignoring the
            # request, since there's no way to honor it.
            raise ValueError(
                "Anthropic has no native schema-less JSON response format; "
                "use JSON_SCHEMA with an explicit json_schema instead"
            )
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            output_config["format"] = {
                "type": "json_schema",
                "schema": config.json_schema,
            }

    if output_config:
        params["output_config"] = output_config

    params.update(config.extra_params)
    return params


def from_native_response(response: Message, model: str) -> LLMResponse:
    tool_calls = [
        ToolCall(id=block.id, name=block.name, arguments=block.input)
        for block in response.content
        if block.type == "tool_use"
    ]

    content = "".join(block.text for block in response.content if block.type == "text")

    usage = response.usage

    return LLMResponse(
        message=AssistantMessage(content=content, tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )
