from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_ollama import ChatOllama

from agent_platform.core.schemas import model_schema
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig
from agent_platform.integrations.credentials.ollama import OllamaCredentials
from agent_platform.core.credentials import (
    resolve_timeout,
)
from agent_platform.core.interfaces.llm.response import ResponseFormat
from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class OllamaLLM(LangChainLLMProvider[OllamaCredentials, OllamaGenerationConfig]):
    def __init__(self, credentials: OllamaCredentials | None = None) -> None:
        super().__init__(
            credentials if credentials is not None else OllamaCredentials()
        )

    def _tool_to_schema(self, tool: "Tool") -> dict[str, Any]:
        schema = model_schema(tool.input_schema)
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
                "required": [],
            },
        }

    def _client(self, config: OllamaGenerationConfig) -> ChatOllama:

        return ChatOllama(
            model=config.model,
            base_url=self._credentials.base_url,
            **_to_langchain_ollama(config, self._credentials),
        )

    def _default_config(self) -> OllamaGenerationConfig:
        return OllamaGenerationConfig()


def _to_langchain_ollama(
    config: OllamaGenerationConfig,
    credentials: OllamaCredentials,
) -> dict[str, Any]:
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

    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["timeout"] = timeout

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
