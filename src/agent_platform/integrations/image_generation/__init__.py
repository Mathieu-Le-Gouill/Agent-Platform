from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "DallEImageGenerator": "agent_platform.integrations.image_generation.dalle.provider",
    "MidjourneyGenerator": "agent_platform.integrations.image_generation.midjourney.provider",
    "StableDiffusionGenerator": "agent_platform.integrations.image_generation.stable_diffusion.provider",
}

# Short slugs for `Settings.default_image_model`'s "<provider>:<model>" strings
# (see `core.config.parse_model_string`), resolved by `config.container.build_provider`.
PROVIDER_ALIASES: dict[str, str] = {
    "dalle": "DallEImageGenerator",
    "midjourney": "MidjourneyGenerator",
    "stable_diffusion": "StableDiffusionGenerator",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
