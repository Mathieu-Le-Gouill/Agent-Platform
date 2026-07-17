from __future__ import annotations
from agent_platform.core.interfaces.vad.config import VADConfig

from pydantic import Field


class TenVadConfig(VADConfig):
    hop_size: int = Field(default=256, ge=64, le=1024)
    threshold: float = 0.5
