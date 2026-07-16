from __future__ import annotations
from agent_platform.core.interfaces.llm.config import GenerationConfig


class MistralGenerationConfig(GenerationConfig):
    model: str = "mistral-medium"
