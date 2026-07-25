from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "WhisperXSTT": "agent_platform.integrations.speech_to_text.whisperx.provider",
    "FasterWhisperSTT": "agent_platform.integrations.speech_to_text.faster_whisper.provider",
    "DeepgramSTT": "agent_platform.integrations.speech_to_text.deepgram.provider",
    "OpenAIWhisperSTT": "agent_platform.integrations.speech_to_text.openai.provider",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
