from __future__ import annotations

from types import ModuleType
from typing import Any

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.config.settings import Settings
from agent_platform.core.errors import ConfigError


def build_provider(domain_module: ModuleType, provider_name: str) -> Any:
    """Instantiate a provider class re-exported by an `integrations.<domain>` module.

    `provider_name` is the class name as exposed by that module's lazy
    `__getattr__` (e.g. "OpenAILLM", "ChromaStore"), already the single
    source of truth for what providers exist in a domain, so no separate
    provider-name registry is kept here. Works for any domain module that
    follows that convention (see `integrations/README.md`), not just `llm`.
    """
    try:
        provider_cls = getattr(domain_module, provider_name)
    except AttributeError as exc:
        raise ConfigError(
            f"Unknown provider {provider_name!r} for {domain_module.__name__!r}"
        ) from exc
    return provider_cls()


def build_agent(settings: Settings) -> ConversationAgent:
    import agent_platform.integrations.llm as llm_module

    return ConversationAgent(
        name=settings.agent_name,
        llm=build_provider(llm_module, settings.llm_provider),
        system_prompt=settings.agent_system_prompt,
        model=settings.llm_model,
        max_iterations=settings.max_iterations,
    )


__all__ = ["build_agent", "build_provider"]
