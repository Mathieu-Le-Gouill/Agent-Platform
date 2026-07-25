from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "PILImageLoader": "agent_platform.integrations.loader.strategies.pil.pil",
    "PyAVLoader": "agent_platform.integrations.loader.strategies.pyav.pyav",
    "SoundFileLoader": (
        "agent_platform.integrations.loader.strategies.soundfile.soundfile"
    ),
    "UnstructuredFileLoader": (
        "agent_platform.integrations.loader.strategies.unstructured.unstructured"
    ),
    "AutoLoader": "agent_platform.integrations.loader.composite.auto",
    "MultiLoader": "agent_platform.integrations.loader.composite.multi",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
