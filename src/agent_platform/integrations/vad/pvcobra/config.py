from __future__ import annotations

from pydantic import Field

from agent_platform.core.interfaces.vad.config import VADConfig


class PvcobraVadConfig(VADConfig):
    # Hardware device selector for the Cobra engine (e.g. "best", "gpu", "cpu:<n>"). https://picovoice.ai/docs/api/cobra-python/
    device: str | None = Field(default=None)
    # Custom path to a Cobra dynamic library for the target platform, overriding the bundled default. https://picovoice.ai/docs/api/cobra-python/
    library_path: str | None = Field(default=None)
    # Voice probability threshold above which a frame counts as speech. https://picovoice.ai/docs/api/cobra-python/
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
