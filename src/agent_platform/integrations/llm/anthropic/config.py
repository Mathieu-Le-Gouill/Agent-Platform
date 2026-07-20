from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.llm.config import GenerationConfig


class AnthropicGenerationConfig(GenerationConfig):
    model: str = "claude-sonnet-4-6"
    """Anthropic model identifier, e.g. "claude-sonnet-4-6" or "claude-opus-4-7"."""

    thinking: bool = False
    """Enable legacy manual extended thinking (`budget_tokens`-style). Deprecated on
    newer models (Opus 4.7+, Sonnet 5) in favor of `effort`."""

    thinking_budget: int | None = None
    """Token budget for legacy `thinking`. Must be strictly less than the effective
    `max_tokens`, or the API returns a 400."""

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    """Modern `output_config.effort` reasoning-depth control; preferred over `thinking`/
    `thinking_budget` on current models."""

    cache_control: bool = False
    """Mark the last eligible content block as an ephemeral prompt-cache breakpoint."""


"""
sources: https://docs.claude.com/en/docs/about-claude/models/overview
         https://docs.claude.com/en/docs/build-with-claude/extended-thinking
         https://docs.claude.com/en/docs/build-with-claude/prompt-caching
"""
