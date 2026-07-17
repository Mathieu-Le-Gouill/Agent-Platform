from __future__ import annotations
from pydantic import Field
from agent_platform.core.interfaces.vad.config import VADConfig


class PvcobraVadConfig(VADConfig):
    device: str | None = Field(default=None)
    library_path: str | None = Field(default=None)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
