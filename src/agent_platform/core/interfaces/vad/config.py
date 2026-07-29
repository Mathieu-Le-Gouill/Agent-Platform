from agent_platform.core.config import ProviderConfig


class VADConfig(ProviderConfig):
    # Audio sample rate (Hz) providers expect. https://github.com/snakers4/silero-vad
    sample_rate: int = 16000
    # Padding added to both sides of a detected speech segment, in ms. https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py (get_speech_timestamps speech_pad_ms)
    speech_pad_ms: int = 30
    # Minimum trailing silence before a speech segment is closed, in ms. https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py (get_speech_timestamps min_silence_duration_ms)
    min_silence_duration_ms: int = 100
    # Minimum accumulated speech duration for a segment to be kept, in ms. https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py (get_speech_timestamps min_speech_duration_ms)
    min_speech_duration_ms: int = 250
