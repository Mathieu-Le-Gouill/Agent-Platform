from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "SileroVAD": "agent_platform.integrations.vad.silero.silero",
    "Webrtcvad": "agent_platform.integrations.vad.webrtc.webrtc",
    "PvcobraVAD": "agent_platform.integrations.vad.pvcobra.pvcobra",
    "TenVAD": "agent_platform.integrations.vad.ten.ten",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
