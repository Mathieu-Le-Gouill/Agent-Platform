from __future__ import annotations

import base64
from typing import Any
from uuid import uuid4

from ollama import ChatResponse

from agent_platform.core.errors import ProviderError
from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    ResponseFormat,
)
from agent_platform.core.schemas.message import (
    AssistantMessage,
    AudioBlock,
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
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig

__all__ = [
    "from_native_response",
    "message_content",
    "to_native_messages",
    "to_native_params",
]


def message_content(m: ContentMessage) -> tuple[str, list[str]]:
    if isinstance(m.content, str):
        return m.content, []

    text_parts: list[str] = []
    images: list[str] = []
    for block in m.blocks:
        match block:
            case TextBlock():
                text_parts.append(block.text)
            case ImageBlock():
                if isinstance(block.image, str):
                    images.append(block.image)
                else:
                    images.append(base64.b64encode(block.image.content).decode())
            case AudioBlock():
                raise ProviderError("Ollama does not support audio content blocks")

    return "".join(text_parts), images


def to_native_messages(prompt: Prompt) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for m in prompt.messages:
        match m:
            case SystemMessage():
                content, images = message_content(m)
                entry: dict[str, Any] = {"role": "system", "content": content}
                if images:
                    entry["images"] = images
                result.append(entry)
            case UserMessage():
                content, images = message_content(m)
                entry = {"role": "user", "content": content}
                if images:
                    entry["images"] = images
                result.append(entry)
            case AssistantMessage():
                content, images = message_content(m)
                entry = {"role": "assistant", "content": content}
                if images:
                    entry["images"] = images
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {"function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in m.tool_calls
                    ]
                result.append(entry)
            case ToolMessage():
                result.append(
                    {
                        "role": "tool",
                        "content": m.result.content,
                        "tool_name": m.result.name,
                    }
                )
    return result


def to_native_params(config: OllamaGenerationConfig) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": config.temperature}
    if config.max_tokens is not None:
        options["num_predict"] = config.max_tokens
    if config.top_p is not None:
        options["top_p"] = config.top_p
    if config.top_k is not None:
        options["top_k"] = config.top_k
    if config.seed is not None:
        options["seed"] = config.seed
    if config.stop_sequences:
        options["stop"] = config.stop_sequences
    if config.repeat_penalty is not None:
        options["repeat_penalty"] = config.repeat_penalty
    if config.mirostat is not None:
        options["mirostat"] = config.mirostat
    if config.mirostat_tau is not None:
        options["mirostat_tau"] = config.mirostat_tau
    if config.mirostat_eta is not None:
        options["mirostat_eta"] = config.mirostat_eta
    if config.num_ctx is not None:
        options["num_ctx"] = config.num_ctx

    params: dict[str, Any] = {"options": options}

    match config.response_format:
        case ResponseFormat.JSON:
            params["format"] = "json"
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            params["format"] = config.json_schema

    params.update(config.extra_params)
    return params


def from_native_response(response: ChatResponse, model: str) -> LLMResponse:
    message = response.message

    tool_calls = [
        ToolCall(
            id=uuid4().hex,
            name=tc.function.name,
            arguments=dict(tc.function.arguments),
        )
        for tc in (message.tool_calls or [])
    ]

    return LLMResponse(
        message=AssistantMessage(content=message.content or "", tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )
