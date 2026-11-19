from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, cast

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.agents.tools.generate_image.tool import GenerateImageTool
from agent_platform.agents.tools.mcp.discovery import discover_mcp_tools
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.agents.tools.transcribe.tool import TranscribeTool
from agent_platform.components.ingestion.speech_to_text.component import SpeechToText
from agent_platform.config.model_string import parse_model_string
from agent_platform.config.settings import Settings
from agent_platform.core.errors import ConfigError
from agent_platform.core.interfaces.image_generation.config import ImageGenConfig
from agent_platform.core.interfaces.llm.fallback import (
    FallbackEntry,
    FallbackLLMProvider,
)
from agent_platform.core.interfaces.mcp.base import BaseMCPClient
from agent_platform.core.interfaces.speech.config import SpeechConfig
from agent_platform.core.interfaces.tracing.base import BaseTracer
from agent_platform.core.interfaces.tracing.noop import NoOpTracer
from agent_platform.core.resilience import RateLimiter
from agent_platform.core.token_usage import TokenUsageAggregator

logger = logging.getLogger(__name__)

# `Settings.tracing_provider` value that turns tracing off outright, whatever
# backend is installed.
_TRACING_DISABLED = "none"


def build_provider(
    domain_module: ModuleType, provider_name: str, **init_kwargs: Any
) -> Any:
    """Instantiate a provider class re-exported by an `integrations.<domain>` module.

    `provider_name` is resolved against that module's `PROVIDER_ALIASES` map first
    (a short slug, e.g. "openai" -> "OpenAILLM"), falling back to `provider_name`
    itself so an explicit class name still works. The domain's lazy `__getattr__`
    is the single source of truth for what providers exist, so no separate
    provider-name registry is kept here. Works for any domain module that follows
    that convention (see `integrations/README.md`), not just `llm`.
    `init_kwargs` go to the provider's constructor.
    """
    aliases: dict[str, str] = getattr(domain_module, "PROVIDER_ALIASES", {})
    class_name = aliases.get(provider_name, provider_name)
    try:
        provider_cls = getattr(domain_module, class_name)
    except AttributeError as exc:
        raise ConfigError(
            f"Unknown provider {provider_name!r} for {domain_module.__name__!r}"
        ) from exc
    return provider_cls(**init_kwargs)


def build_provider_from_model_string(
    domain_module: ModuleType, model_string: str
) -> tuple[Any, str]:
    """Resolve a `"<provider>:<model>"` setting into `(provider_instance, model)`.

    See `config.model_string.parse_model_string` for the string format.
    """
    provider_name, model = parse_model_string(model_string)
    return build_provider(domain_module, provider_name), model


def _validate_llm_options(
    llm_module: ModuleType, llm_options: dict[str, dict[str, Any]]
) -> None:
    """Check every `llm_options` entry against its provider's config class.

    Every entry is checked, whether the chain uses its provider or not, so a
    typo fails at startup instead of when a fallback is first reached. Only
    the slug and the field name appear in an error, never an option value.
    """
    aliases: dict[str, str] = llm_module.PROVIDER_ALIASES
    for slug, options in llm_options.items():
        if slug not in aliases:
            raise ConfigError(
                f"Unknown LLM provider {slug!r} in llm_options, "
                f"expected one of {sorted(aliases)}"
            )
        getattr(llm_module, aliases[slug]).validate_options(options)


def _build_llm_entry(
    llm_module: ModuleType, model_string: str, llm_options: dict[str, dict[str, Any]]
) -> tuple[Any, str]:
    """Build one chain entry bound to its model and to its provider's `llm_options`."""
    provider_name, model = parse_model_string(model_string)
    class_name = llm_module.PROVIDER_ALIASES.get(provider_name, provider_name)
    slug = next(
        (s for s, name in llm_module.PROVIDER_ALIASES.items() if name == class_name),
        provider_name,
    )
    provider = build_provider(
        llm_module, provider_name, model=model, options=llm_options.get(slug)
    )
    return provider, model


def build_llm_with_fallback(settings: Settings) -> tuple[Any, str]:
    """Resolve `settings.default_llm_model` (+ `fallback_llm_models`) into a
    single `BaseLLMProvider`.

    Each provider is bound to its model string's model and to its
    `settings.llm_options` entry. With no fallbacks configured, returns the
    primary provider directly (the common case stays a plain single-provider
    call, no wrapper overhead). Otherwise wraps the chain in a
    `FallbackLLMProvider`, which owns retrying the next entry when an earlier
    one's circuit breaker trips - callers see one `BaseLLMProvider`, not the
    chain behind it.
    """
    import agent_platform.integrations.llm as llm_module

    _validate_llm_options(llm_module, settings.llm_options)
    primary, primary_model = _build_llm_entry(
        llm_module, settings.default_llm_model, settings.llm_options
    )
    if not settings.fallback_llm_models:
        return primary, primary_model

    entries = [FallbackEntry(provider=primary, model=primary_model)]
    for model_string in settings.fallback_llm_models:
        provider, model = _build_llm_entry(
            llm_module, model_string, settings.llm_options
        )
        entries.append(FallbackEntry(provider=provider, model=model))

    rate_limiter = (
        RateLimiter(rate=settings.llm_rate_limit, burst=settings.llm_rate_limit_burst)
        if settings.llm_rate_limit is not None
        else None
    )
    return FallbackLLMProvider(entries, rate_limiter=rate_limiter), primary_model


def build_tracer(settings: Settings) -> BaseTracer:
    """Resolve `settings.tracing_provider` into the tracer to install process-wide.

    Falls back to the no-op tracer when the backend's SDK isn't installed, so
    a deployment that never added the `tracing` extra still starts instead of
    failing on an import it doesn't need.
    """
    if settings.tracing_provider == _TRACING_DISABLED:
        return NoOpTracer()

    import agent_platform.integrations.tracing as tracing_module

    try:
        tracer = build_provider(tracing_module, settings.tracing_provider)
    except ImportError:
        logger.warning(
            "Tracing provider %r is unavailable (install the matching extra); "
            "spans will be dropped",
            settings.tracing_provider,
        )
        return NoOpTracer()
    return cast(BaseTracer, tracer)


def build_agent(settings: Settings) -> ConversationAgent:
    import agent_platform.integrations.image_generation as image_module
    import agent_platform.integrations.speech_to_text as speech_module

    llm, llm_model = build_llm_with_fallback(settings)
    image_generator, image_model = build_provider_from_model_string(
        image_module, settings.default_image_model
    )
    speech_to_text, audio_model = build_provider_from_model_string(
        speech_module, settings.default_audio_model
    )

    registry = ToolRegistry()
    registry.register(
        GenerateImageTool(
            generator=image_generator,
            default_config=ImageGenConfig(model=image_model),
        )
    )
    registry.register(
        TranscribeTool(
            speech_to_text=SpeechToText(backend=speech_to_text),
            default_config=SpeechConfig(model=audio_model),
        )
    )

    return ConversationAgent(
        name=settings.agent_name,
        llm=llm,
        tool_registry=registry,
        system_prompt=settings.agent_system_prompt,
        model=llm_model,
        max_iterations=settings.max_iterations,
        max_history_turns=settings.max_history_turns,
        token_usage_aggregator=TokenUsageAggregator(),
    )


async def _build_mcp_client(command: list[str]) -> BaseMCPClient:
    """Construct a `BaseMCPClient` for a `[command, *args]` entry.

    Imports `StdioMCPClient` lazily, same reason `build_agent` imports
    `integrations.llm`/etc. inside the function rather than at module scope:
    it lives behind the optional `tools-mcp` extra (it's the one file that
    imports the third-party `mcp` SDK, per `integrations/README.md`'s "only
    integrations/ imports third-party packages" rule), so importing it
    unconditionally at module level would break `build_agent`/this module
    for anyone without that extra installed, even if they never configure
    `mcp_stdio_servers`.
    """
    from agent_platform.integrations.mcp.stdio.provider import StdioMCPClient

    executable, *args = command
    return StdioMCPClient(executable, args)


async def build_agent_async(
    settings: Settings,
) -> tuple[ConversationAgent, list[BaseMCPClient]]:
    """`build_agent`, plus MCP tools discovered from `settings.mcp_stdio_servers`.

    A separate entrypoint rather than making `build_agent` itself async:
    connecting to an MCP server is an inherently async handshake (subprocess
    launch + initialize), unlike every other provider `build_agent` resolves
    synchronously, so forcing that on every caller (evals, scripts, most of
    the test suite) would be a needless breaking change for the common case
    of zero configured MCP servers.

    Returns the connected clients alongside the agent so the caller (see
    `api/app.py`'s lifespan) can `aclose()` each one at shutdown - an MCP
    server is a stateful session (`BaseMCPClient`'s docstring), not a
    fire-and-forget call, so something has to own closing it.
    """
    agent = build_agent(settings)

    clients: list[BaseMCPClient] = []
    try:
        for command in settings.mcp_stdio_servers.values():
            client = await _build_mcp_client(command)
            # A client whose own `connect()` raises cleans up after itself, so
            # it joins `clients` only once connected.
            await client.connect()
            clients.append(client)
            for tool in await discover_mcp_tools(client):
                agent.tool_registry.register(tool, timeout=settings.mcp_tool_timeout)
    except BaseException:
        await _close_clients(clients)
        raise

    return agent, clients


async def _close_clients(clients: list[BaseMCPClient]) -> None:
    """Close every client, logging (not raising) a failing `aclose()` so the
    error that triggered the cleanup is the one the caller sees."""
    for client in clients:
        try:
            await client.aclose()
        except Exception:
            logger.exception("Closing an MCP client failed during startup cleanup")


__all__ = [
    "build_agent",
    "build_agent_async",
    "build_llm_with_fallback",
    "build_provider",
    "build_provider_from_model_string",
    "build_tracer",
]
