from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any

from opentelemetry.trace import Span

from agent_platform.core.schemas.token import TokenUsage
from agent_platform.core.tracing import traced_span

__all__ = [
    "GenAIAttributes",
    "record_token_usage",
    "traced_operation_span",
]


class GenAIAttributes:
    """OTel GenAI semantic-convention attribute keys.

    https://github.com/open-telemetry/semantic-conventions-genai (the gen-ai
    conventions moved out of the main semantic-conventions repo; the old
    opentelemetry.io/docs/specs/semconv/gen-ai/ page just redirects there
    now) - kept as constants so every call site (agent loop, tool calls, LLM
    calls, and whatever gets instrumented next: embeddings, reranking, ...)
    spells them identically. All `gen_ai.*` keys below are still
    "Development" stability, so upstream can rename them again.
    """

    OPERATION_NAME = "gen_ai.operation.name"
    PROVIDER_NAME = "gen_ai.provider.name"
    REQUEST_MODEL = "gen_ai.request.model"
    AGENT_NAME = "gen_ai.agent.name"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    BATCH_ID = "gen_ai.batch.id"
    CONVERSATION_ID = "gen_ai.conversation.id"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"


def record_token_usage(span: Span, usage: TokenUsage) -> None:
    """Record `usage` on `span` using the GenAI token-usage attribute keys."""
    span.set_attribute(GenAIAttributes.USAGE_INPUT_TOKENS, usage.input_tokens)
    span.set_attribute(GenAIAttributes.USAGE_OUTPUT_TOKENS, usage.output_tokens)


def traced_operation_span(
    operation: str, /, attributes: Mapping[str, Any] | None = None
) -> AbstractContextManager[Span]:
    """`traced_span` for a GenAI operation: names the span `operation` and sets
    `gen_ai.operation.name` to match, since every current call site (`chat`,
    `invoke_agent`, `execute_tool`) needs that pairing and any future one
    (embeddings, reranking, ...) will too.
    """
    return traced_span(
        operation, {GenAIAttributes.OPERATION_NAME: operation, **(attributes or {})}
    )
