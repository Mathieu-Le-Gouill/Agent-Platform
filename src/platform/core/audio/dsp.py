from __future__ import annotations

import torchaudio

from models.audio import AudioSegment
from platform.core.audio.io import AudioSegmentIO


class AudioSegmentDSP:

    # --- Tools ---

    @staticmethod
    def resample(segment: AudioSegment, target_rate: int) -> AudioSegment:
        if segment.sample_rate == target_rate:
            return segment
        tensor = AudioSegmentIO.to_tensor(segment).unsqueeze(0)
        resampled = torchaudio.functional.resample(
            tensor, segment.sample_rate, target_rate
        ).squeeze(0)
        return AudioSegmentIO.from_tensor(resampled, target_rate)
    

    @staticmethod
    def chunk(segment: AudioSegment, seconds: float) -> list[AudioSegment]:
        arr = AudioSegmentIO.to_numpy(segment)
        size = int(seconds * segment.sample_rate)
        return [
            AudioSegment(
                data=arr[i : i + size].tobytes(),
                sample_rate=segment.sample_rate,
                channels=segment.channels,
                start_ms=int(i / segment.sample_rate * 1000),
                end_ms=int(min(i + size, len(arr)) / segment.sample_rate * 1000),
            )
            for i in range(0, len(arr), size)
        ]