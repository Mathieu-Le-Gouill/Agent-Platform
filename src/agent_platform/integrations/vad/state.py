from dataclasses import dataclass


@dataclass(slots=True)
class VADState:
    in_speech: bool = False

    segment_start: int | None = None
    last_speech_end: int | None = None

    speech_ms: int = 0
    silence_ms: int = 0
