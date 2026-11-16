from __future__ import annotations

import base64
import json
from typing import Any

from openai.types.chat import ChatCompletion

from agent_platform.core.interfaces.llm.batch import BatchJob, BatchStatus
from agent_platform.core.interfaces.llm.response import (
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
from agent_platform.integrations.llm._finish_reason import to_finish_reason
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig

__all__ = [
    "batch_job_from_native",
    "block_to_native",
    "content_to_native",
    "from_native_response",
    "is_reasoning_model",
    "strict_json_schema",
    "to_native_messages",
    "to_native_params",
]

# Reasoning models (o-series, gpt-5 non-chat) reject non-default temperature/top_p
# and only accept reasoning_effort on this family. https://platform.openai.com/docs/guides/reasoning
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4-mini")

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


class _NotStrictError(Exception):
    """A schema uses something OpenAI strict mode does not accept."""


# A property schema must say what it is, or strict mode rejects it.
_TYPING_KEYS = ("type", "$ref", "anyOf", "enum")
_SCHEMA_MAP_KEYS = ("properties", "$defs", "definitions")
_SCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")


def _strictify(schema: dict[str, Any], *, top: bool) -> dict[str, Any]:
    """Return a copy of `schema` with `additionalProperties: false` on every
    object; raise `_NotStrictError` on anything strict mode rejects."""
    out = dict(schema)

    is_object = out.get("type") == "object" or "properties" in out
    if is_object:
        if out.get("additionalProperties", False) is not False:
            raise _NotStrictError
        if "properties" not in out and not top:
            raise _NotStrictError
        properties = out.get("properties", {})
        if not set(properties) <= set(out.get("required", [])):
            raise _NotStrictError
        if "required" not in out:
            out["required"] = []
        out["additionalProperties"] = False

    for key in _SCHEMA_MAP_KEYS:
        if key in out:
            out[key] = {
                name: _strictify(sub, top=False) for name, sub in out[key].items()
            }
    for key in _SCHEMA_LIST_KEYS:
        if key in out:
            out[key] = [_strictify(sub, top=False) for sub in out[key]]
    if isinstance(out.get("items"), dict):
        out["items"] = _strictify(out["items"], top=False)
    # OpenAI rejects a missing `items` and does not document an empty one
    # (a bare `list` or `list[Any]`); sending non-strict cannot fail.
    if out.get("type") == "array":
        items = out.get("items")
        if not isinstance(items, dict) or not any(k in items for k in _TYPING_KEYS):
            raise _NotStrictError

    for sub in out.get("properties", {}).values():
        if not any(k in sub for k in _TYPING_KEYS):
            raise _NotStrictError
    return out


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any] | None:
    """Return a copy of `schema` that OpenAI strict mode accepts, or `None`.

    Strict mode needs `additionalProperties: false` on every object, every
    property listed in `required`, and every property and array item typed.
    The copy sets the
    first; a schema that breaks the others cannot be made strict without
    changing what the model may return (a defaulted field would become
    mandatory), so it gets `None` and the caller sends it non-strict. `schema`
    is never mutated: `model_schema` hands out a cached dict.
    """
    try:
        return _strictify(schema, top=True)
    except _NotStrictError:
        return None


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
    if not reasoning_model and config.temperature is not None:
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
            strict_schema = (
                strict_json_schema(config.json_schema) if config.strict else None
            )
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": strict_schema or config.json_schema,
                    "strict": strict_schema is not None,
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
        finish_reason=to_finish_reason(response.choices[0].finish_reason),
        response_id=response.id,
    )
