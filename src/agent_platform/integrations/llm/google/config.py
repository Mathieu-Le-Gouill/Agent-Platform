from __future__ import annotations

from agent_platform.core.interfaces.llm.config import GenerationConfig


class GoogleGenerationConfig(GenerationConfig):
    # `model` is a Gemini model id such as "gemini-2.5-flash".

    # Extended-thinking token budget; wins over the generic `reasoning_effort`
    # when both are set. `None` leaves the model's own default in place.
    thinking_budget: int | None = None
    # Whether to surface the model's thinking trace as part of the response.
    include_thoughts: bool = False
