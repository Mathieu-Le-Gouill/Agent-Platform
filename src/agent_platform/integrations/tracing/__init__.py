from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "LangfuseTracer": "agent_platform.integrations.tracing.langfuse.provider",
}

# Short slugs for `Settings.tracing_provider`, resolved by
# `config.container.build_provider`.
PROVIDER_ALIASES: dict[str, str] = {
    "langfuse": "LangfuseTracer",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
