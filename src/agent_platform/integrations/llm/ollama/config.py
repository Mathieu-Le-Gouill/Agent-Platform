from __future__ import annotations
from agent_platform.core.interfaces.llm.config import GenerationConfig


class OllamaGenerationConfig(GenerationConfig):
    model: str = "llama3.2"
