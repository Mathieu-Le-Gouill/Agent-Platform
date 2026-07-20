from __future__ import annotations
from pydantic import Field
from agent_platform.core.interfaces.vad.config import VADConfig


class WebrtcVadConfig(VADConfig):
    # Filtering aggressiveness, 0 (least aggressive/most permissive) to 3 (most aggressive at filtering non-speech). https://github.com/wiseman/py-webrtcvad
    mode: int = Field(default=1, ge=0, le=3)
