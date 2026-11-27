from __future__ import annotations

import base64
import logging
from typing import Any

from anthropic import transform_schema
from anthropic.types import Message

from agent_platform.core.errors import ProviderError
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
from agent_platform.integrations.llm._generic_fields import (
    levels_for_model,
    nearest_level,
)
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig

__all__ = [
    "batch_job_from_native",
    "block_to_native",
    "content_to_native",
    "from_native_response",
    "to_native_messages",
    "to_native_params",
]

logger = logging.getLogger(__name__)

# Used when the config sets no `max_tokens` (the API requires one). 1024
# truncated ordinary answers. 4096 is the output cap of the smallest
# Claude 3 model still in `_SAMPLING_MODEL_PREFIXES`, and at a typical
# 50-80 tokens/s it finishes inside `DEFAULT_LLM_TIMEOUT_SECONDS` (120 s);
# a larger default would time out non-streaming calls, and each timeout is
# retried and billed again. Set `max_tokens` for longer outputs.
_DEFAULT_MAX_TOKENS = 4096

# Models that still accept `temperature`, `top_p` and `top_k` on the API
# version the SDK sends (2023-06-01). Claude Opus 4.7 and every later model
# reject non-default values with a 400, so a model matching none of these
# prefixes, a future one included, gets none of them.
# https://platform.claude.com/docs/en/build-with-claude/thinking
# ("Sampling parameters")
_SAMPLING_MODEL_PREFIXES = (
    "claude-3",
    "claude-4-",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-2025",
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-0",
    "claude-sonnet-4-2025",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
)
# Every Claude 4+ model rejects `temperature` and `top_p` in one request;
# among the allowlisted models only Claude 3 accepts both. Anthropic's
# model-migration guide ("Sampling parameters: temperature OR top_p, not
# both") and docs.claude.com/en/docs/about-claude/models/migrating-to-claude-4.
_BOTH_SAMPLING_MODEL_PREFIXES = ("claude-3",)
# Thinking modes under which those models reject `temperature` and `top_k`.
_THINKING_ON_TYPES = frozenset({"enabled", "adaptive"})

# `output_config.effort` support per model.
# https://platform.claude.com/docs/en/build-with-claude/effort ("supportedModels"
# and the `xhigh` row). Each prefix names one released model family, so a
# future model id matches none and gets the requested level unchanged.
_NO_EFFORT_MODEL_PREFIXES = (
    "claude-3",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-2025",
    "claude-sonnet-4-0",
    "claude-sonnet-4-2025",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
)
_LEVELS_WITHOUT_XHIGH = frozenset({"low", "medium", "high"})
_EFFORT_LEVELS: dict[str, frozenset[str]] = {
    "claude-opus-4-6": _LEVELS_WITHOUT_XHIGH,
    "claude-sonnet-4-6": _LEVELS_WITHOUT_XHIGH,
    "claude-opus-4-5": _LEVELS_WITHOUT_XHIGH,
    "claude-mythos-preview": _LEVELS_WITHOUT_XHIGH,
}
_DEFAULT_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh"})

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


def _effort(config: AnthropicGenerationConfig) -> str | None:
    """Returns the `output_config.effort` value: the provider field, else the generic level.

    The provider field is sent unchecked. The generic level is clamped to the
    levels the model accepts; a model without effort support gets none.
    """
    if config.effort is not None:
        return config.effort
    level = config.reasoning_effort
    if level is None:
        return None
    model = config.model
    if model.lower().startswith(_NO_EFFORT_MODEL_PREFIXES):
        logger.warning("reasoning_effort is ignored: %r does not support effort", model)
        return None
    supported = levels_for_model(model, _EFFORT_LEVELS) or _DEFAULT_EFFORT_LEVELS
    return nearest_level("anthropic", level, supported, model)


def _sampling_params(config: AnthropicGenerationConfig) -> dict[str, Any]:
    """Returns the sampling parameters the model accepts; each one dropped is logged.

    Only the parameter names and the model id are logged, never the values.
    """
    model = config.model
    params = {
        name: value
        for name, value in (
            ("temperature", config.temperature),
            ("top_p", config.top_p),
            ("top_k", config.top_k),
        )
        if value is not None
    }

    def drop(name: str, reason: str) -> None:
        if params.pop(name, None) is not None:
            logger.warning("%s is ignored: %s", name, reason)

    model_lower = model.lower()
    if not model_lower.startswith(_SAMPLING_MODEL_PREFIXES):
        for name in list(params):
            drop(name, f"{model!r} does not accept sampling parameters")
        return params
    thinking = config.extra_params.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") in _THINKING_ON_TYPES:
        drop("temperature", "it is incompatible with thinking")
        drop("top_k", "it is incompatible with thinking")
    if "temperature" in params and not model_lower.startswith(
        _BOTH_SAMPLING_MODEL_PREFIXES
    ):
        drop("top_p", f"{model!r} rejects temperature and top_p together")
    return params


def _output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Returns `schema` reduced to the JSON Schema subset structured outputs accept.

    `anthropic.transform_schema` closes every object and moves unsupported
    constraints (`minimum`, `maxLength`, ...) into descriptions. A schema it
    cannot read (a node with no `type`, such as an `Any` field) is sent as
    is, and the API names what it rejects.
    """
    try:
        return transform_schema(schema)
    except ValueError:
        return schema


def to_native_params(
    config: AnthropicGenerationConfig, *, has_tools: bool = False
) -> dict[str, Any]:
    params: dict[str, Any] = _sampling_params(config)
    if config.stop_sequences:
        params["stop_sequences"] = config.stop_sequences
    params["max_tokens"] = config.max_tokens or _DEFAULT_MAX_TOKENS
    if config.parallel_tool_calls is not None and has_tools:
        params["tool_choice"] = {
            "type": "auto",
            "disable_parallel_tool_use": not config.parallel_tool_calls,
        }

    output_config: dict[str, Any] = {}
    effort = _effort(config)
    if effort is not None:
        output_config["effort"] = effort

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
                "schema": _output_schema(config.json_schema),
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
        finish_reason=to_finish_reason(response.stop_reason),
        response_id=response.id,
    )
