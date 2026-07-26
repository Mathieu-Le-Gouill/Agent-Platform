from __future__ import annotations

from agent_platform.core.interfaces.llm.config import GenerationConfig


class GoogleGenerationConfig(GenerationConfig):
    model: str = "gemini-2.5-flash"
    # Extended-thinking token budget; provider-specific, mirrors Anthropic's
    # `thinking_budget`. `None` leaves the model's own default in place.
    thinking_budget: int | None = None
    # Whether to surface the model's thinking trace as part of the response.
    include_thoughts: bool = False
