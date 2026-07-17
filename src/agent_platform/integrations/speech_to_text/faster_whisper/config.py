from __future__ import annotations
from agent_platform.core.interfaces.speech.config import SpeechConfig


class FasterWhisperConfig(SpeechConfig):
    model_size: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "float32"
    beam_size: int = 5
    language: str | None = None
    vad_filter: bool = True
    min_duration_ms: int = 5000
