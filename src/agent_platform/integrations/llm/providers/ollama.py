from langchain_ollama import ChatOllama
from typing import Any

from agent_platform.integrations.llm.config import GenerationConfig, ResponseFormat
from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider


class OllamaLLM(LangChainLLMProvider):
    def _client(
        self,
        model: str,
        config: GenerationConfig | None,
    ) -> ChatOllama:
        cfg = config or GenerationConfig()
        return ChatOllama(
            model=model,
            **_to_langchain_ollama(cfg),
        )


def _to_langchain_ollama(config: GenerationConfig | None) -> dict[str, Any]:
    if config is None:
        return {}

    params: dict[str, Any] = {"temperature": config.temperature}
    if config.max_tokens is not None:
        params["num_predict"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.top_k is not None:
        params["top_k"] = config.top_k
    if config.seed is not None:
        params["seed"] = config.seed
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.frequency_penalty is not None:
        params["repeat_penalty"] = config.frequency_penalty
    if config.timeout is not None:
        params["timeout"] = config.timeout

    match config.response_format:
        case ResponseFormat.JSON:
            params["format"] = "json"
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            params["format"] = config.json_schema

    return params
