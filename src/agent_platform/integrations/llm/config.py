from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from agent_platform.integrations.llm.response import ResponseFormat


class GenerationConfig(BaseModel):

    model_config = ConfigDict(populate_by_name=True)

    model: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] = []
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    timeout: float | None = None
    max_retries: int | None = None
    response_format: ResponseFormat = ResponseFormat.TEXT
    json_schema: dict | None = None


class AnthropicGenerationConfig(GenerationConfig):
    model: str = "claude-sonnet-4-6"
    thinking: bool = False
    thinking_budget: int | None = None
    cache_control: bool = False


class OpenAIGenerationConfig(GenerationConfig):
    model: str = "gpt-4.1"
    reasoning_effort: str | None = None
    parallel_tool_calls: bool = True


class MistralGenerationConfig(GenerationConfig):
    model: str = "mistral-medium"


class OllamaGenerationConfig(GenerationConfig):
    model: str = "llama3.2"


class HuggingFaceGenerationConfig(GenerationConfig):
    task: str = "text-generation"
    repo_id: str = "deepseek-ai/DeepSeek-R1-0528"
    device: str | None = None
    provider: str = "auto"