from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "PILImageLoader": "agent_platform.integrations.loader.strategies.pil.provider",
    "PyAVLoader": "agent_platform.integrations.loader.strategies.pyav.provider",
    "SoundFileLoader": (
        "agent_platform.integrations.loader.strategies.soundfile.provider"
    ),
    "UnstructuredFileLoader": (
        "agent_platform.integrations.loader.strategies.unstructured.provider"
    ),
    "AutoLoader": "agent_platform.integrations.loader.composite.auto",
    "MultiLoader": "agent_platform.integrations.loader.composite.multi",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
