from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from agent_platform.core.interfaces.llm.config import GenerationConfig


class OllamaGenerationConfig(GenerationConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    model: str = "llama3.2"
