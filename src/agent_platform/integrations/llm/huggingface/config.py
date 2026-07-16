from __future__ import annotations
from agent_platform.core.interfaces.llm.config import GenerationConfig


class HuggingFaceGenerationConfig(GenerationConfig):
    task: str = "text-generation"
    repo_id: str = "deepseek-ai/DeepSeek-R1-0528"
    device: str | None = None
    provider: str = "auto"
