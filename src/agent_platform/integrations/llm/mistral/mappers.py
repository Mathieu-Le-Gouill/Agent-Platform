from __future__ import annotations

import base64
import json
from typing import Any
from uuid import uuid4

from mistralai.models import ChatCompletionResponse

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
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig

__all__ = [
    "block_to_native",
    "content_to_native",
    "from_native_response",
    "to_native_messages",
    "to_native_params",
]


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
            return {"type": "image_url", "image_url": url}
        case AudioBlock():
            data = base64.b64encode(block.audio.content).decode()
            return {"type": "audio", "input_audio": data}


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


def to_native_params(config: MistralGenerationConfig) -> dict[str, Any]:
    params: dict[str, Any] = {"temperature": config.temperature}
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
                    "schema": config.json_schema,
                    "name": config.json_schema_name,
                    "strict": config.json_schema_strict,
                },
            }

    params.update(config.extra_params)
    return params


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
        finish_reason=FinishReason.STOP,
    )
