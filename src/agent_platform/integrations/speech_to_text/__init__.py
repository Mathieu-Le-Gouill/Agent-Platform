from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "WhisperXSTT": "agent_platform.integrations.speech_to_text.whisperx.whisperx",
    "FasterWhisperSTT": "agent_platform.integrations.speech_to_text.faster_whisper.faster_whisper",
    "DeepgramSTT": "agent_platform.integrations.speech_to_text.deepgram.deepgram",
    "OpenAIWhisperSTT": "agent_platform.integrations.speech_to_text.openai.openai",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
