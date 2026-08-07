from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "OTelTracer": "agent_platform.integrations.tracing.otel.provider",
}

# Short slugs for `Settings.tracing_provider`, resolved by
# `config.container.build_provider`.
PROVIDER_ALIASES: dict[str, str] = {
    "otel": "OTelTracer",
    "opentelemetry": "OTelTracer",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
