from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.speech.config import SpeechConfig


class OpenAIWhisperConfig(SpeechConfig):
    model: str = (
        "whisper-1"  # transcription model id, e.g. "whisper-1", "gpt-4o-transcribe"
    )
    language: str | None = (
        None  # ISO-639-1 language code; None lets the model auto-detect
    )
    temperature: float = (
        0.0  # sampling temperature; higher values increase output randomness
    )
    min_duration_ms: int = (
        5000  # minimum audio duration required before transcription runs
    )
    # None auto-selects per model: verbose_json for whisper-1, json for gpt-4o(-mini)-transcribe.
    response_format: Literal["json", "text", "verbose_json"] | None = None
    # whisper-1 + verbose_json only.
    timestamp_granularities: list[Literal["word", "segment"]] | None = None
    prompt: str | None = (
        None  # optional text to bias/guide the model's style or vocabulary
    )


# sources: https://platform.openai.com/docs/api-reference/audio/createTranscription
