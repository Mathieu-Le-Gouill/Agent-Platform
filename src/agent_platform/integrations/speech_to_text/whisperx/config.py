from __future__ import annotations
from agent_platform.core.interfaces.speech.config import SpeechConfig


class WhisperXConfig(SpeechConfig):
    model_size: str = "large-v3"
    device: str = "cpu"
    compute_type: str = "float32"
    batch_size: int = 16
    min_duration_ms: int = 5000
