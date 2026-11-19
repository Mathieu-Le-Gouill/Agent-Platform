from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.llm.config import GenerationConfig


class AnthropicGenerationConfig(GenerationConfig):
    """Anthropic config. `model` is an Anthropic model id such as "claude-sonnet-4-6"."""

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    """`output_config.effort` reasoning-depth control. Wins over the generic
    `reasoning_effort` when both are set; use it for `max`, which the generic
    levels do not have."""

    cache_control: bool = False
    """Mark the last eligible content block as an ephemeral prompt-cache breakpoint."""


"""
sources: https://platform.claude.com/docs/en/about-claude/models/overview
         https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         https://platform.claude.com/docs/en/build-with-claude/prompt-caching
"""
