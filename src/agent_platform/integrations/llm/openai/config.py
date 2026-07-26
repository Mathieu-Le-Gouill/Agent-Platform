from __future__ import annotations

from agent_platform.core.interfaces.llm.config import GenerationConfig


class OpenAIGenerationConfig(GenerationConfig):
    model: str = "gpt-4.1"
    """OpenAI model identifier, e.g. "gpt-4.1" or "o4-mini"."""

    reasoning_effort: str | None = None
    """Reasoning depth for reasoning models only (o-series, gpt-5 family); rejected by
    non-reasoning models."""

    parallel_tool_calls: bool = True
    """Whether the model may return multiple tool calls in a single turn."""

    strict: bool = True
    """Enforce strict JSON Schema conformance for the JSON_SCHEMA response format."""


"""
sources: https://developers.openai.com/api/docs/models
         https://developers.openai.com/api/docs/guides/reasoning
         https://developers.openai.com/api/docs/guides/function-calling
         https://developers.openai.com/api/docs/guides/structured-outputs
         https://developers.openai.com/api/reference/resources/chat (exact request schema; ChatOpenAI uses /v1/chat/completions, not /v1/responses)
"""
