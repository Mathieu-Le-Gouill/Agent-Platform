from __future__ import annotations
from agent_platform.core.interfaces.speech.config import SpeechConfig


class OpenAIWhisperConfig(SpeechConfig):
    model: str = "whisper-1"
    language: str | None = None
    temperature: float = 0.0
    min_duration_ms: int = 5000
