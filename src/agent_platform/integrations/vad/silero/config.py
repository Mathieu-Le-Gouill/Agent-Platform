from __future__ import annotations

from agent_platform.core.interfaces.vad.config import VADConfig


class SileroVadConfig(VADConfig):
    # Speech probability threshold above which a frame counts as speech. https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py (get_speech_timestamps threshold)
    threshold: float = 0.5
    # Maximum duration of a single speech segment, in seconds; inf disables the cap. https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py (get_speech_timestamps max_speech_duration_s)
    max_speech_duration_s: float = float("inf")
