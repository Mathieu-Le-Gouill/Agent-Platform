from __future__ import annotations

import base64
import json
from typing import Any

from openai.types.chat import ChatCompletion

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
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig

__all__ = [
    "batch_job_from_native",
    "block_to_native",
    "content_to_native",
    "from_native_response",
    "is_reasoning_model",
    "to_native_messages",
    "to_native_params",
]

# Reasoning models (o-series, gpt-5 non-chat) reject non-default temperature/top_p
# and only accept reasoning_effort on this family. https://platform.openai.com/docs/guides/reasoning
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4-mini")
_DEFAULT_TEMPERATURE: float = OpenAIGenerationConfig.model_fields["temperature"].default

_BATCH_STATUS_MAP: dict[str, BatchStatus] = {
    "validating": BatchStatus.PENDING,
    "in_progress": BatchStatus.IN_PROGRESS,
    "finalizing": BatchStatus.IN_PROGRESS,
    "cancelling": BatchStatus.IN_PROGRESS,
    "completed": BatchStatus.COMPLETED,
    "failed": BatchStatus.FAILED,
    "expired": BatchStatus.EXPIRED,
    "cancelled": BatchStatus.CANCELLED,
}


def is_reasoning_model(model: str) -> bool:
    model_lower = model.lower()
    if model_lower.startswith(_REASONING_MODEL_PREFIXES):
        return True
    return model_lower.startswith("gpt-5") and "chat" not in model_lower


def batch_job_from_native(batch: Any) -> BatchJob:
    counts = getattr(batch, "request_counts", None)
    return BatchJob(
        id=batch.id,
        status=_BATCH_STATUS_MAP.get(batch.status, BatchStatus.PENDING),
        request_count=counts.total if counts is not None else None,
        completed_count=counts.completed if counts is not None else None,
    )


def block_to_native(block: ContentBlock) -> dict[str, Any]:
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


def content_to_native(m: ContentMessage) -> str | list[dict[str, Any]]:
    if isinstance(m.content, str):
        return m.content
    return [block_to_native(b) for b in m.blocks]


def to_native_messages(prompt: Prompt) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for m in prompt.messages:
        match m:
            case SystemMessage():
                result.append({"role": "system", "content": content_to_native(m)})
            case UserMessage():
                result.append({"role": "user", "content": content_to_native(m)})
            case AssistantMessage():
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": content_to_native(m),
                }
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                result.append(entry)
            case ToolMessage():
                result.append(
                    {
                        "role": "tool",
                        "content": m.result.content,
                        "tool_call_id": m.result.tool_call_id,
                    }
                )
    return result


def to_native_params(config: OpenAIGenerationConfig) -> dict[str, Any]:
    reasoning_model = is_reasoning_model(config.model)

    params: dict[str, Any] = {}
    if not reasoning_model and config.temperature != _DEFAULT_TEMPERATURE:
        params["temperature"] = config.temperature
    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.seed is not None:
        params["seed"] = config.seed

    if config.frequency_penalty is not None:
        params["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        params["presence_penalty"] = config.presence_penalty

    if config.reasoning_effort is not None and reasoning_model:
        params["reasoning_effort"] = config.reasoning_effort
    if not config.parallel_tool_calls:
        params["parallel_tool_calls"] = config.parallel_tool_calls

    match config.response_format:
        case ResponseFormat.JSON:
            params["response_format"] = {"type": "json_object"}
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": config.json_schema,
                    "strict": config.strict,
                },
            }

    params.update(config.extra_params)
    return params


def from_native_response(response: ChatCompletion, model: str) -> LLMResponse:
    message = response.choices[0].message

    tool_calls = [
        # `_tool_to_schema` only ever sends `"type": "function"` tools, so the
        # response only ever carries function-tool calls, never the SDK's
        # custom-tool-call variant.
        ToolCall(
            id=tc.id,
            name=tc.function.name,  # type: ignore[union-attr]
            arguments=json.loads(tc.function.arguments),  # type: ignore[union-attr]
        )
        for tc in (message.tool_calls or [])
    ]

    usage = response.usage

    return LLMResponse(
        message=AssistantMessage(content=message.content or "", tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )
