from __future__ import annotations
from agent_platform.core.interfaces.vad.config import VADConfig


class SileroVadConfig(VADConfig):
    threshold: float = 0.5
