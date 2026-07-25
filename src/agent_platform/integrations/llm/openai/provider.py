from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI

from agent_platform.core.credentials import (
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import require_secret
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.core.schemas import model_schema
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class OpenAILLM(LangChainLLMProvider[OpenAIGenerationConfig]):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, OpenAICredentials)

    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        schema = model_schema(tool.input_schema)
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
            "strict": True,
        }

    def _client(self, config: OpenAIGenerationConfig) -> ChatOpenAI:
        api_key = require_secret(
            self._credentials.api_key,
            "OPENAI API key is required but was not provided",
        )

        return ChatOpenAI(
            model=config.model,
            api_key=api_key,
            base_url=self._credentials.base_url,
            **_to_langchain_openai(config, self._credentials),
        )

    def _default_config(self) -> OpenAIGenerationConfig:
        return OpenAIGenerationConfig()


# Reasoning models (o-series, gpt-5 non-chat) reject non-default temperature/top_p
# and only accept reasoning_effort on this family. https://platform.openai.com/docs/guides/reasoning
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4-mini")
_DEFAULT_TEMPERATURE: float = OpenAIGenerationConfig.model_fields["temperature"].default


def _is_reasoning_model(model: str) -> bool:
    model_lower = model.lower()
    if model_lower.startswith(_REASONING_MODEL_PREFIXES):
        return True
    return model_lower.startswith("gpt-5") and "chat" not in model_lower


def _to_langchain_openai(
    config: OpenAIGenerationConfig,
    credentials: OpenAICredentials,
) -> dict[str, Any]:
    reasoning_model = _is_reasoning_model(config.model)

    params: dict[str, Any] = {}
    if not reasoning_model and config.temperature != _DEFAULT_TEMPERATURE:
        params["temperature"] = config.temperature
    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.seed is not None:
        params["seed"] = config.seed

    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["timeout"] = timeout
    params["max_retries"] = resolve_max_retries(config.max_retries, credentials)

    model_kwargs: dict[str, Any] = {}
    if config.frequency_penalty is not None:
        model_kwargs["frequency_penalty"] = config.frequency_penalty
    if config.presence_penalty is not None:
        model_kwargs["presence_penalty"] = config.presence_penalty

    if config.reasoning_effort is not None and reasoning_model:
        model_kwargs["reasoning_effort"] = config.reasoning_effort
    if not config.parallel_tool_calls:
        model_kwargs["parallel_tool_calls"] = config.parallel_tool_calls

    match config.response_format:
        case ResponseFormat.JSON:
            model_kwargs["response_format"] = {"type": "json_object"}
        case ResponseFormat.JSON_SCHEMA:
            if config.json_schema is None:
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            model_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": config.json_schema,
                    "strict": config.strict,
                },
            }

    if model_kwargs:
        params["model_kwargs"] = model_kwargs

    params.update(config.extra_params)
    return params
