from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, model_validator


class ScoreKind(StrEnum):
    SIMILARITY = "similarity"
    RELEVANCE = "relevance"
    CONFIDENCE = "confidence"
    QUALITY = "quality"
    SENTIMENT = "sentiment"


class Score(BaseModel, frozen=True):
    value: float
    kind: ScoreKind = ScoreKind.SIMILARITY
    low: float = 0.0
    high: float = 1.0

    @model_validator(mode="after")
    def _validate_bounds(self) -> Score:
        if self.low >= self.high:
            raise ValueError(
                f"Score bounds invalid: low={self.low} >= high={self.high}"
            )
        if not (self.low <= self.value <= self.high):
            raise ValueError(
                f"{self.kind.value} score {self.value} outside [{self.low}, {self.high}]"
            )
        return self

    @classmethod
    def similarity(cls, value: float, *, low: float = 0.0, high: float = 1.0) -> Score:
        return cls(value=value, kind=ScoreKind.SIMILARITY, low=low, high=high)

    @classmethod
    def relevance(cls, value: float) -> Score:
        return cls(value=value, kind=ScoreKind.RELEVANCE)

    @classmethod
    def confidence(cls, value: float) -> Score:
        return cls(value=value, kind=ScoreKind.CONFIDENCE)

    @classmethod
    def quality(cls, value: float) -> Score:
        return cls(value=value, kind=ScoreKind.QUALITY)

    @classmethod
    def logit(cls, value: float, kind: ScoreKind = ScoreKind.CONFIDENCE) -> Score:
        # unbounded wrapper (not an actual logit transform) for raw provider scores that may fall outside [0, 1]
        return cls(value=value, kind=kind, low=float("-inf"), high=float("inf"))

    @property
    def normalized(self) -> float:
        span = self.high - self.low
        if span == 0 or span == float("inf"):
            return self.value
        return (self.value - self.low) / span

    def to_percentage(self) -> float:
        return round(self.normalized * 100, 2)

    def exceeds(self, threshold: float) -> bool:
        return self.value > threshold

    def is_high_confidence(self, threshold: float = 0.85) -> bool:
        return self.normalized > threshold
