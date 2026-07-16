from __future__ import annotations
from agent_platform.core.interfaces.llm.config import GenerationConfig


class OpenAIGenerationConfig(GenerationConfig):
    model: str = "gpt-4.1"
    reasoning_effort: str | None = None
    parallel_tool_calls: bool = True
