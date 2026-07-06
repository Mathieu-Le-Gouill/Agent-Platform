from langchain_mistralai import ChatMistralAI
from pydantic import SecretStr
from typing import Any

from agent_platform.integrations.llm.config import GenerationConfig, ResponseFormat
from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider


class MistralLLM(LangChainLLMProvider):

    def __init__(self, api_key: SecretStr) -> None:
        self._api_key = api_key

    def _client(
        self,
        model: str,
        config: GenerationConfig | None,
    ) -> ChatMistralAI:
        cfg = config or GenerationConfig()
        return ChatMistralAI(
            model_name=model,
            api_key=self._api_key,
            **_to_langchain_mistral(cfg),
        )


def _to_langchain_mistral(config: GenerationConfig | None) -> dict[str, Any]:
    if config is None:
        return {}

    params: dict[str, Any] = {"temperature": config.temperature}
    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.seed is not None:
        params["random_seed"] = config.seed
    if config.timeout is not None:
        params["timeout"] = config.timeout
    if config.max_retries is not None:
        params["max_retries"] = config.max_retries

    model_kwargs: dict[str, Any] = {}
    if config.frequency_penalty is not None:
        model_kwargs["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        model_kwargs["presence_penalty"] = config.presence_penalty

    match config.response_format:
        case ResponseFormat.JSON:
            model_kwargs["response_format"] = {"type": "json_object"}
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError("json_schema is required for JSON_SCHEMA response format")
            model_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": config.json_schema,
            }

    if model_kwargs:
        params["model_kwargs"] = model_kwargs
    return params