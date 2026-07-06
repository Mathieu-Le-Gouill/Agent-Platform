from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from openai._types import Omit
from typing import TypeVar, Any

from agent_platform.integrations.llm.config import GenerationConfig, OpenAIConfig, ResponseFormat
from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider


T = TypeVar("T")


def _omit_none(value: T | None) -> T | Omit:
    return value if value is not None else Omit()


class OpenAILLM(LangChainLLMProvider):

    def __init__(self, api_key: SecretStr) -> None:
        self._api_key = api_key

    def _client(
        self,
        model: str,
        config: GenerationConfig | None,
    ) -> ChatOpenAI:
        cfg = config or GenerationConfig()
        return ChatOpenAI(
            model=model,
            api_key=self._api_key,
            **_to_langchain_openai(cfg),
        )


def _to_langchain_openai(config: GenerationConfig | None) -> dict[str, Any]:
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
        params["seed"] = config.seed
    if config.timeout is not None:
        params["timeout"] = config.timeout
    if config.max_retries is not None:
        params["max_retries"] = config.max_retries

    model_kwargs: dict[str, Any] = {}
    if config.frequency_penalty is not None:
        model_kwargs["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        model_kwargs["presence_penalty"] = config.presence_penalty

    if isinstance(config, OpenAIConfig):
        if config.reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = config.reasoning_effort
        if not config.parallel_tool_calls:
            model_kwargs["parallel_tool_calls"] = config.parallel_tool_calls

    match config.response_format:
        case ResponseFormat.JSON:
            model_kwargs["response_format"] = {"type": "json_object"}
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError("json_schema is required for JSON_SCHEMA response format")
            model_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": config.json_schema},
            }

    if model_kwargs:
        params["model_kwargs"] = model_kwargs
    return params

