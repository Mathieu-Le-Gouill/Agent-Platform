from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioSegment:

    data: bytes
    sample_rate: int
    channels: int = 1
    start_ms: int | None = None 
    end_ms: int | None = None

    # --- Properties ---

    @property
    def duration_ms(self) -> float:
        # PCM: num_frames = len(bytes) / (channels * bytes_per_sample)
        # For 32-bit float PCM: bytes_per_sample = 4
        num_frames = len(self.data) / (self.channels * 4)
        return (num_frames / self.sample_rate) * 1000

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AudioSegment):
            return NotImplemented
        # Two segments are the same if their content and format match
        return (
            self.data == other.data
            and self.sample_rate == other.sample_rate
            and self.channels == other.channels
        )

    def __hash__(self) -> int:
        return hash((self.data, self.sample_rate, self.channels))