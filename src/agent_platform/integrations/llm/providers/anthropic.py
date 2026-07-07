from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr

from agent_platform.core.schema import model_schema
from agent_platform.integrations.llm.config import GenerationConfig, AnthropicConfig
from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class AnthropicLLM(LangChainLLMProvider):
    def __init__(self, api_key: SecretStr) -> None:
        self._api_key = api_key

    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        schema = model_schema(tool.input_schema)
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": schema,
        }

    def _client(
        self,
        model: str,
        config: GenerationConfig | None,
    ) -> ChatAnthropic:
        cfg = config or GenerationConfig()
        return ChatAnthropic(
            model_name=model,
            api_key=self._api_key,
            **_to_langchain_anthropic(cfg),
        )


def _to_langchain_anthropic(config: GenerationConfig | None) -> dict[str, Any]:
    if config is None:
        return {"max_tokens": 1024}

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
    if config.timeout is not None:
        params["timeout"] = config.timeout
    if config.max_retries is not None:
        params["max_retries"] = config.max_retries

    if isinstance(config, AnthropicConfig):
        if config.thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.thinking_budget or 5000,
            }
        if config.cache_control:
            params["cache_control"] = {"type": "ephemeral"}

    return params
