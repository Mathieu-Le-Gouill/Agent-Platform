from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class VADConfig:
    sample_rate = 16000
    max_samples: int = 50
    speech_pad_ms: int = 30
    min_silence_duration_ms: int = 100
    min_speech_duration_ms: int = 250


@dataclass(slots=True, frozen=True)
class SileroVadConfig(VADConfig):
    threshold: float = 0.5


@dataclass(slots=True, frozen=True)
class WebrtcVadConfig(VADConfig):
    mode: int = 1


@dataclass(slots=True, frozen=True)
class PvcobraVadConfig(VADConfig):
    threshold: float = 0.5


@dataclass(slots=True, frozen=True)
class TenVadConfig(VADConfig):
    hop_size = 256 | 160
    threshold: float = 0.5
