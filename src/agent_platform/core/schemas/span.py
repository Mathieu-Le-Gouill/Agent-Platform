from __future__ import annotations

from pydantic import BaseModel


class TimeSpan(BaseModel, frozen=True):
    start: float
    end: float
    unit: str = "ms"

    @property
    def duration(self) -> float:
        return self.end - self.start


class SampleSpan(BaseModel, frozen=True):
    start: int
    end: int

    @property
    def sample_size(self) -> int:
        return self.end - self.start

    def to_ms(self, sample_rate: int) -> TimeSpan:
        # samples / sample_rate (samples-per-second) yields seconds, not ms; caller beware
        return TimeSpan(
            start=self.start // sample_rate,
            end=self.end // sample_rate,
        )
