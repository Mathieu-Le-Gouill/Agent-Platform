from pydantic import BaseModel, Field, ConfigDict


class VADConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    sample_rate: int = 16000
    max_samples: int = 50
    speech_pad_ms: int = 30
    min_silence_duration_ms: int = 100
    min_speech_duration_ms: int = 250
