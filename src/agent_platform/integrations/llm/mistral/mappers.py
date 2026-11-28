from __future__ import annotations

import base64
import json
from typing import Any
from uuid import uuid4

from mistralai.models import ChatCompletionResponse

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
from agent_platform.integrations.llm._generic_fields import warn_unsupported
from agent_platform.integrations.llm._json_schema import json_schema_name
from agent_platform.integrations.llm._media import image_data_url
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig

__all__ = [
    "block_to_native",
    "content_to_native",
    "error_body",
    "from_native_response",
    "to_native_messages",
    "to_native_params",
]


def block_to_native(block: ContentBlock) -> dict[str, Any]:
    match block:
        case TextBlock():
            return {"type": "text", "text": block.text}
        case ImageBlock():
            url = (
                block.image
                if isinstance(block.image, str)
                else image_data_url(block.image)
            )
            return {"type": "image_url", "image_url": url}
        case AudioBlock():
            data = base64.b64encode(block.audio.content).decode()
            return {"type": "input_audio", "input_audio": data}


def error_body(body: str) -> dict[str, Any]:
    """Returns the JSON object of an error response body, or an empty dict.

    Mistral's error body carries `message`, `type` and a four-digit `code`
    string; some responses (a 401) carry only `detail`, others no JSON at all.
    """
    try:
        parsed = json.loads(body)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
                        "name": m.result.name,
                    }
                )
    return result


def to_native_params(
    config: MistralGenerationConfig, *, has_tools: bool = False
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if config.reasoning_effort is not None:
        warn_unsupported("mistral", "reasoning_effort")
    if config.parallel_tool_calls is not None and has_tools:
        params["parallel_tool_calls"] = config.parallel_tool_calls
    if config.temperature is not None:
        params["temperature"] = config.temperature
    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.seed is not None:
        params["random_seed"] = config.seed

    if config.frequency_penalty is not None:
        params["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        params["presence_penalty"] = config.presence_penalty

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
                    "schema": _closed_schema(config.json_schema),
                    "name": json_schema_name(config.json_schema),
                    "strict": True,
                },
            }

    params.update(config.extra_params)
    return params


def _closed_schema(node: Any) -> Any:
    """Return a copy of a JSON schema with `additionalProperties: false` on every object.

    Mistral's strict mode expects closed objects: its SDK helper
    `mistralai.extra.response_format_from_pydantic_model` applies the same
    rule. Unlike OpenAI's strict mode, fields with a default stay optional,
    so `required` is left as is. An object that already sets
    `additionalProperties` (a `dict[str, X]` field) keeps its value schema.
    """
    if isinstance(node, dict):
        closed = {key: _closed_schema(value) for key, value in node.items()}
        if closed.get("type") == "object":
            closed.setdefault("additionalProperties", False)
        return closed
    if isinstance(node, list):
        return [_closed_schema(value) for value in node]
    return node


def from_native_response(response: ChatCompletionResponse, model: str) -> LLMResponse:
    message = response.choices[0].message

    tool_calls = [
        ToolCall(
            # Mistral's SDK types `tc.id` as optional, but the API always assigns
            # one; the uuid4 fallback only guards against that type/reality gap.
            id=tc.id or uuid4().hex,
            name=tc.function.name,
            arguments=(
                tc.function.arguments
                if isinstance(tc.function.arguments, dict)
                else json.loads(tc.function.arguments)
            ),
        )
        for tc in (message.tool_calls or [])
    ]

    content = message.content if isinstance(message.content, str) else ""
    usage = response.usage

    return LLMResponse(
        message=AssistantMessage(content=content or "", tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=(usage.prompt_tokens or 0) if usage else 0,
            output_tokens=(usage.completion_tokens or 0) if usage else 0,
        ),
        model=model,
        finish_reason=to_finish_reason(response.choices[0].finish_reason),
        response_id=response.id,
    )
