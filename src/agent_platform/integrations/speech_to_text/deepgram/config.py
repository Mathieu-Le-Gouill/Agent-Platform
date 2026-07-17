from __future__ import annotations
from agent_platform.core.interfaces.speech.config import SpeechConfig


class DeepgramConfig(SpeechConfig):
    model: str = "nova-2"
    smart_format: bool = True
    language: str | None = None
