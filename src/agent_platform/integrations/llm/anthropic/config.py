from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.llm.config import GenerationConfig


class AnthropicGenerationConfig(GenerationConfig):
    model: str = "claude-sonnet-4-6"
    """Anthropic model identifier, e.g. "claude-sonnet-4-6" or "claude-opus-4-7"."""

    thinking: bool = False
    """Enable legacy manual extended thinking (`budget_tokens`-style). Deprecated
    starting with the Claude 4.6 generation (incl. the default model above; still
    accepted there) and rejected outright with a 400 starting at Claude 4.7+, in
    favor of `effort`."""

    thinking_budget: int | None = None
    """Token budget for legacy `thinking`. Must be strictly less than the effective
    `max_tokens`, or the API returns a 400."""

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    """Modern `output_config.effort` reasoning-depth control; preferred over `thinking`/
    `thinking_budget` on current models."""

    cache_control: bool = False
    """Mark the last eligible content block as an ephemeral prompt-cache breakpoint."""


"""
sources: https://platform.claude.com/docs/en/about-claude/models/overview
         https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         https://platform.claude.com/docs/en/build-with-claude/prompt-caching
"""
