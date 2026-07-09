from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_mistralai import ChatMistralAI

from agent_platform.core.schema import model_schema
from agent_platform.integrations.llm.config import MistralGenerationConfig
from agent_platform.integrations.credentials import (
    MistralCredentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.integrations.llm.response import ResponseFormat
from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider
from agent_platform.core.errors import MissingCredentialError

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class MistralLLM(LangChainLLMProvider[MistralCredentials, MistralGenerationConfig]):
    def __init__(self, credentials: MistralCredentials | None = None) -> None:
        super().__init__(credentials if credentials is not None else MistralCredentials())

    def _tool_to_schema(self, tool: "Tool") -> dict[str, Any]:
        schema = model_schema(tool.input_schema)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            },
        }

    def _client(self, config: MistralGenerationConfig) -> ChatMistralAI:

        if self._credentials.api_key is None:
            raise MissingCredentialError("Mistral API key is required but was not provided")

        return ChatMistralAI(
            model_name=config.model,
            api_key=self._credentials.api_key,
            base_url=self._credentials.base_url,

            **_to_langchain_mistral(config, self._credentials),
        )
    
    def _default_config(self) -> MistralGenerationConfig: 
        return MistralGenerationConfig()


def _to_langchain_mistral(
    config: MistralGenerationConfig,
    credentials: MistralCredentials,
) -> dict[str, Any]:
    params: dict[str, Any] = {"temperature": config.temperature}
    if config.max_tokens is not None:
        params["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.stop_sequences:
        params["stop"] = config.stop_sequences
    if config.seed is not None:
        params["random_seed"] = config.seed

    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["timeout"] = int(timeout)
    params["max_retries"] = resolve_max_retries(config.max_retries, credentials)

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
                raise ValueError(
                    "json_schema is required for JSON_SCHEMA response format"
                )
            model_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": config.json_schema,
            }

    if model_kwargs:
        params["model_kwargs"] = model_kwargs
    return params
