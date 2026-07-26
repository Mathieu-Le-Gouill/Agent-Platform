from __future__ import annotations

from agent_platform.core.interfaces.speech.config import SpeechConfig


class DeepgramConfig(SpeechConfig):
    model: str = "nova-2"  # Deepgram model name, see model doc
    smart_format: bool = (
        True  # applies punctuation/casing/formatting, see smart-format doc
    )
    language: str | None = (
        None  # BCP-47 language code; None uses the model's default language, see language doc
    )
    detect_language: bool = (
        False  # when True (and `language` unset), asks Deepgram to auto-detect the
        # language on pre-recorded transcription; not supported for streaming
    )
    diarize: bool = False  # tags each word with a speaker label, see diarization doc
    punctuate: bool = True  # adds punctuation and capitalization, see punctuation doc


"""
sources: https://developers.deepgram.com/docs/model
         https://developers.deepgram.com/docs/smart-format
         https://developers.deepgram.com/docs/language
         https://developers.deepgram.com/docs/diarization
         https://developers.deepgram.com/docs/punctuation
"""
