from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "OpenAIModeration": "agent_platform.integrations.moderation.openai.provider",
    "LocalModeration": "agent_platform.integrations.moderation.local.provider",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
