from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeSpan:
    start: float
    end: float
    unit: str = "ms"


    @property
    def duration(self) -> float:
        return self.end - self.start
    


@dataclass(frozen=True, slots=True)
class SampleSpan:
    start: int
    end: int

    @property
    def sample_size(self) -> int:
        return self.end - self.start
    

    def to_ms(self, sample_rate: int) -> TimeSpan:
        return TimeSpan(
            start=self.start // sample_rate,
            end=self.end // sample_rate,
        )