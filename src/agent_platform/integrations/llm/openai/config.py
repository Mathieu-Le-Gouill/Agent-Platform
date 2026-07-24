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
sources: https://platform.openai.com/docs/models
         https://platform.openai.com/docs/guides/reasoning
         https://platform.openai.com/docs/guides/function-calling
         https://platform.openai.com/docs/guides/structured-outputs
"""
