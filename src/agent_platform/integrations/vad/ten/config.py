from __future__ import annotations
from typing import Literal
from agent_platform.core.interfaces.vad.config import VADConfig

from pydantic import Field


class TenVadConfig(VADConfig):
    # Samples processed per call; only 160 and 256 (at 16kHz) are supported/optimized. https://github.com/TEN-framework/ten-vad
    hop_size: Literal[160, 256] = Field(default=256)
    # Speech probability threshold above which a frame counts as speech. https://github.com/TEN-framework/ten-vad
    threshold: float = 0.5
