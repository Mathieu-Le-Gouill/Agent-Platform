from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_openai import ChatOpenAI

from agent_platform.core.schemas import model_schema
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig
from agent_platform.integrations.credentials.openai import OpenAICredentials
from agent_platform.core.credentials import (
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider
from agent_platform.core.errors import MissingCredentialError

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class OpenAILLM(LangChainLLMProvider[OpenAICredentials, OpenAIGenerationConfig]):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        super().__init__(
            credentials if credentials is not None else OpenAICredentials()
        )

    def _tool_to_schema(self, tool: "Tool") -> dict[str, Any]:
        schema = model_schema(tool.input_schema)
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
            "strict": True,
        }

    def _client(self, config: OpenAIGenerationConfig) -> ChatOpenAI:

        if self._credentials.api_key is None:
            raise MissingCredentialError(
                "OPENAI API key is required but was not provided"
            )

        return ChatOpenAI(
            model=config.model,
            api_key=self._credentials.api_key,
            base_url=self._credentials.base_url,
            **_to_langchain_openai(config, self._credentials),
        )

    def _default_config(self) -> OpenAIGenerationConfig:
        return OpenAIGenerationConfig()


def _to_langchain_openai(
    config: OpenAIGenerationConfig,
    credentials: OpenAICredentials,
) -> dict[str, Any]:
    params: dict[str, Any] = {"temperature": config.temperature}
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

    if config.reasoning_effort is not None:
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
                "json_schema": {"name": "response", "schema": config.json_schema},
            }

    if model_kwargs:
        params["model_kwargs"] = model_kwargs

    params.update(config.extra_params)
    return params
