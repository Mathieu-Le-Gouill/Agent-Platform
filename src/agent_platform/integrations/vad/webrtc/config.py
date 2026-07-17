from __future__ import annotations
from pydantic import Field
from agent_platform.core.interfaces.vad.config import VADConfig


class WebrtcVadConfig(VADConfig):
    mode: int = Field(default=1, ge=0, le=3)
    aggressiveness: int = 3
