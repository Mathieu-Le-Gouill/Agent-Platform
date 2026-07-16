from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_anthropic import ChatAnthropic

from agent_platform.core.schemas import model_schema
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.credentials.anthropic import AnthropicCredentials
from agent_platform.core.credentials import (
    resolve_max_retries,
    resolve_timeout,
)

from agent_platform.core.errors import MissingCredentialError

from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class AnthropicLLM(
    LangChainLLMProvider[AnthropicCredentials, AnthropicGenerationConfig]
):
    def __init__(self, credentials: AnthropicCredentials | None = None) -> None:
        super().__init__(
            credentials if credentials is not None else AnthropicCredentials()
        )

    def _tool_to_schema(self, tool: "Tool") -> dict[str, Any]:
        schema = model_schema(tool.input_schema)
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": schema,
            "required": [],
        }

    def _client(self, config: AnthropicGenerationConfig) -> ChatAnthropic:
        if self._credentials.api_key is None:
            raise MissingCredentialError(
                "Anthropic API key is required but was not provided"
            )

        return ChatAnthropic(
            model_name=config.model,
            api_key=self._credentials.api_key,
            base_url=self._credentials.base_url,
            **_to_langchain_anthropic(config, self._credentials),
        )

    def _default_config(self) -> AnthropicGenerationConfig:
        return AnthropicGenerationConfig()


def _to_langchain_anthropic(
    config: AnthropicGenerationConfig,
    credentials: AnthropicCredentials,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "max_tokens": config.max_tokens or 1024,
        "temperature": config.temperature,
    }
    if config.top_p is not None:
        params["top_p"] = config.top_p
    if config.top_k is not None:
        params["top_k"] = config.top_k
    if config.stop_sequences:
        params["stop_sequences"] = config.stop_sequences

    timeout = resolve_timeout(config.timeout, credentials)
    if timeout is not None:
        params["timeout"] = timeout
    params["max_retries"] = resolve_max_retries(config.max_retries, credentials)

    if config.thinking:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": config.thinking_budget or 5000,
        }
    if config.cache_control:
        params["cache_control"] = {"type": "ephemeral"}

    params.update(config.extra_params)
    return params
