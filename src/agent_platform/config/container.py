from __future__ import annotations

from typing import Any

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.config.settings import Settings
from agent_platform.core.errors import ConfigError
from agent_platform.core.interfaces.llm.base import BaseLLMProvider

_LLM_PROVIDERS: dict[str, str] = {
    "openai": "OpenAILLM",
    "anthropic": "AnthropicLLM",
    "mistral": "MistralLLM",
    "ollama": "OllamaLLM",
    "huggingface": "HuggingFaceLLM",
}


def _build_llm(provider: str) -> BaseLLMProvider:
    import agent_platform.integrations.llm as llm_module

    class_name = _LLM_PROVIDERS.get(provider)
    if class_name is None:
        raise ConfigError(f"Unknown LLM provider: {provider!r}")
    provider_cls: Any = getattr(llm_module, class_name)
    return provider_cls()


def build_agent(settings: Settings) -> ConversationAgent:
    return ConversationAgent(
        name=settings.agent_name,
        llm=_build_llm(settings.llm_provider),
        system_prompt=settings.agent_system_prompt,
        model=settings.llm_model,
        max_iterations=settings.max_iterations,
    )


__all__ = ["build_agent"]
