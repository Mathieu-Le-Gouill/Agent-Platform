from __future__ import annotations

from types import ModuleType
from typing import Any

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.agents.tools.generate_image import GenerateImageTool
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.agents.tools.transcribe import TranscribeTool
from agent_platform.components.speech_to_text import SpeechToText
from agent_platform.config.model_string import parse_model_string
from agent_platform.config.settings import Settings
from agent_platform.core.errors import ConfigError
from agent_platform.core.interfaces.image_generation.config import ImageGenConfig
from agent_platform.core.interfaces.speech.config import SpeechConfig


def build_provider(domain_module: ModuleType, provider_name: str) -> Any:
    """Instantiate a provider class re-exported by an `integrations.<domain>` module.

    `provider_name` is resolved against that module's `PROVIDER_ALIASES` map first
    (a short slug, e.g. "openai" -> "OpenAILLM"), falling back to `provider_name`
    itself so an explicit class name still works. The domain's lazy `__getattr__`
    is the single source of truth for what providers exist, so no separate
    provider-name registry is kept here. Works for any domain module that follows
    that convention (see `integrations/README.md`), not just `llm`.
    """
    aliases: dict[str, str] = getattr(domain_module, "PROVIDER_ALIASES", {})
    class_name = aliases.get(provider_name, provider_name)
    try:
        provider_cls = getattr(domain_module, class_name)
    except AttributeError as exc:
        raise ConfigError(
            f"Unknown provider {provider_name!r} for {domain_module.__name__!r}"
        ) from exc
    return provider_cls()


def build_provider_from_model_string(
    domain_module: ModuleType, model_string: str
) -> tuple[Any, str]:
    """Resolve a `"<provider>:<model>"` setting into `(provider_instance, model)`.

    See `config.model_string.parse_model_string` for the string format.
    """
    provider_name, model = parse_model_string(model_string)
    return build_provider(domain_module, provider_name), model


def build_agent(settings: Settings) -> ConversationAgent:
    import agent_platform.integrations.image_generation as image_module
    import agent_platform.integrations.llm as llm_module
    import agent_platform.integrations.speech_to_text as speech_module

    llm, llm_model = build_provider_from_model_string(
        llm_module, settings.default_llm_model
    )
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
    )


__all__ = ["build_agent", "build_provider", "build_provider_from_model_string"]
