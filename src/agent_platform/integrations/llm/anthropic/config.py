from __future__ import annotations
from agent_platform.core.interfaces.llm.config import GenerationConfig


class AnthropicGenerationConfig(GenerationConfig):
    model: str = "claude-sonnet-4-6"
    thinking: bool = False
    thinking_budget: int | None = None
    cache_control: bool = False
