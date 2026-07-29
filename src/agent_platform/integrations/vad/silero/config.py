from __future__ import annotations

from agent_platform.core.interfaces.vad.config import VADConfig


class SileroVadConfig(VADConfig):
    # Speech probability threshold above which a frame counts as speech. https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py (get_speech_timestamps threshold)
    threshold: float = 0.5
    # Maximum duration of a single speech segment, in seconds; inf disables the cap. https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py (get_speech_timestamps max_speech_duration_s)
    max_speech_duration_s: float = float("inf")
    # Rolling streaming-buffer cap in samples; ~5s of context at 16kHz. Specific to
    # Silero's own buffer-and-rerun streaming approach; other providers process
    # frame-by-frame via the shared FrameBasedVAD state machine instead.
    # https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
    max_samples: int = 80_000
