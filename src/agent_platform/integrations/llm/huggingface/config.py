from __future__ import annotations
from pydantic import BaseModel, ConfigDict
from agent_platform.core.interfaces.llm.config import GenerationConfig


class HuggingFaceGenerationConfig(GenerationConfig):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    task: str = "text-generation"
    repo_id: str = "deepseek-ai/DeepSeek-R1-0528"
    device: str | None = None
    provider: str = "auto"
