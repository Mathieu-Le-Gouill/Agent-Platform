from __future__ import annotations
from agent_platform.core.interfaces.speech.config import SpeechConfig


class FasterWhisperConfig(SpeechConfig):
    model_size: str = "large-v3"  # Whisper model checkpoint name/size to load
    device: str = "cpu"  # inference device, e.g. "cpu" or "cuda"
    compute_type: str = "float32"  # ctranslate2 quantization/precision, e.g. "float32", "int8"
    beam_size: int = 5  # beam search width used during decoding
    language: str | None = None  # ISO language code; None lets the model auto-detect
    vad_filter: bool = True  # applies voice activity detection to skip silent segments
    min_duration_ms: int = 5000  # minimum audio duration required before transcription runs
    word_timestamps: bool = False  # emits per-word start/end timestamps
    condition_on_previous_text: bool = True  # feeds prior segment text back in as decoding context


# sources: https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
