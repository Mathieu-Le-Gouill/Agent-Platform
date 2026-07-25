from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_anthropic import ChatAnthropic

from agent_platform.core.credentials import (
    resolve_credentials,
    resolve_max_retries,
    resolve_timeout,
)
from agent_platform.core.errors import require_secret
from agent_platform.core.schemas import model_schema
from agent_platform.integrations.credentials import AnthropicCredentials
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider

if TYPE_CHECKING:
    from agent_platform.agents.tools.base import Tool


class AnthropicLLM(LangChainLLMProvider[AnthropicGenerationConfig]):
    def __init__(self, credentials: AnthropicCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, AnthropicCredentials)

    def _tool_to_schema(self, tool: Tool) -> dict[str, Any]:
        schema = model_schema(tool.input_schema)
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": schema,
            "required": [],
        }

    def _client(self, config: AnthropicGenerationConfig) -> ChatAnthropic:
        api_key = require_secret(
            self._credentials.api_key,
            "Anthropic API key is required but was not provided",
        )

        return ChatAnthropic(
            model_name=config.model,
            api_key=api_key,
            base_url=self._credentials.base_url,
            **_to_langchain_anthropic(config, self._credentials),
        )

    def _default_config(self) -> AnthropicGenerationConfig:
        return AnthropicGenerationConfig()


_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_THINKING_BUDGET = 5000
# Higher fallback so the default thinking_budget stays comfortably below
# max_tokens (Anthropic requires budget_tokens < max_tokens or the request 400s).
_DEFAULT_MAX_TOKENS_WITH_THINKING = 8192


def _to_langchain_anthropic(
    config: AnthropicGenerationConfig,
    credentials: AnthropicCredentials,
) -> dict[str, Any]:
    params: dict[str, Any] = {
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

    if config.effort is not None:
        # Modern effort control supersedes manual thinking_budget management;
        # skip the legacy `thinking` payload entirely when set.
        params["max_tokens"] = config.max_tokens or _DEFAULT_MAX_TOKENS
        params["effort"] = config.effort
    elif config.thinking:
        budget_tokens = config.thinking_budget or _DEFAULT_THINKING_BUDGET
        if config.max_tokens is not None:
            max_tokens = config.max_tokens
            if budget_tokens >= max_tokens:
                budget_tokens = max(1, max_tokens - 1024)
        else:
            # No explicit max_tokens: size it to comfortably fit the (possibly
            # user-supplied) budget rather than clamping the user's budget down
            # to an unrelated fixed default.
            max_tokens = max(_DEFAULT_MAX_TOKENS_WITH_THINKING, budget_tokens + 1024)
        params["max_tokens"] = max_tokens
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }
    else:
        params["max_tokens"] = config.max_tokens or _DEFAULT_MAX_TOKENS

    if config.cache_control:
        # `cache_control` is not a ChatAnthropic constructor field; routing it
        # through model_kwargs is the documented pass-through mechanism and
        # avoids the silent "unrecognized kwarg" fallback warning.
        params.setdefault("model_kwargs", {})["cache_control"] = {"type": "ephemeral"}

    params.update(config.extra_params)
    return params
