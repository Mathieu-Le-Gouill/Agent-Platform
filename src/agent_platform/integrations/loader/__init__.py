from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "PILImageLoader": "agent_platform.integrations.loader.strategies.pil.pil",
    "PyAVLoader": "agent_platform.integrations.loader.strategies.pyav.pyav",
    "SoundFileLoader": (
        "agent_platform.integrations.loader.strategies.soundfile.soundfile"
    ),
    "UnstructuredFileLoader": (
        "agent_platform.integrations.loader.strategies.unstructured.unstructured"
    ),
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
