from agent_platform.core.schemas.config import ProviderConfig


class VADConfig(ProviderConfig):
    sample_rate: int = 16000
    max_samples: int = 50
    speech_pad_ms: int = 30
    min_silence_duration_ms: int = 100
    min_speech_duration_ms: int = 250
