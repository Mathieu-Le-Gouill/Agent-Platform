from __future__ import annotations

from agent_platform.core.interfaces.speech.config import SpeechConfig


class WhisperXConfig(SpeechConfig):
    model_size: str = "large-v3"  # Whisper model checkpoint name/size to load
    device: str = "cpu"  # inference device, e.g. "cpu" or "cuda"
    compute_type: str = (
        "float32"  # ctranslate2 quantization/precision, e.g. "float32", "int8"
    )
    batch_size: int = 16  # number of audio segments transcribed per batch
    min_duration_ms: int = (
        5000  # minimum audio duration required before transcription runs
    )
    language: str | None = None  # ISO language code; None lets the model auto-detect
    align: bool = (
        True  # runs phoneme-level forced alignment for accurate word timestamps
    )
    diarize: bool = False  # runs speaker diarization to tag words with speaker labels
    hf_token: str | None = (
        None  # HuggingFace access token required for diarization/alignment models
    )
    min_speakers: int | None = None  # lower bound hint for diarization speaker count
    max_speakers: int | None = None  # upper bound hint for diarization speaker count


# sources: https://github.com/m-bain/whisperX
