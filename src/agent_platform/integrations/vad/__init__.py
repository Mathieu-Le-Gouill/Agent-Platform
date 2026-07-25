from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "SileroVAD": "agent_platform.integrations.vad.silero.provider",
    "Webrtcvad": "agent_platform.integrations.vad.webrtc.provider",
    "PvcobraVAD": "agent_platform.integrations.vad.pvcobra.provider",
    "TenVAD": "agent_platform.integrations.vad.ten.provider",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
