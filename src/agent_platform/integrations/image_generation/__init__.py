from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "DallEImageGenerator": "agent_platform.integrations.image_generation.dalle.dalle",
    "MidjourneyGenerator": "agent_platform.integrations.image_generation.midjourney.midjourney",
    "StableDiffusionGenerator": "agent_platform.integrations.image_generation.stable_diffusion.stable_diffusion",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
