from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from agent_platform.core.interfaces.llm.config import GenerationConfig


class OpenAIGenerationConfig(GenerationConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    model: str = "gpt-4.1"
    reasoning_effort: str | None = None
    parallel_tool_calls: bool = True
