from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "WhisperXSTT": "agent_platform.integrations.speech_to_text.whisperx.whisperx",
    "FasterWhisperSTT": "agent_platform.integrations.speech_to_text.faster_whisper.faster_whisper",
    "DeepgramSTT": "agent_platform.integrations.speech_to_text.deepgram.deepgram",
    "OpenAIWhisperSTT": "agent_platform.integrations.speech_to_text.openai.openai",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
