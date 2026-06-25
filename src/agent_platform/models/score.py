from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ScoreKind(str, Enum):
    SIMILARITY   = "similarity"    # cosine / dot-product from an embedder
    RELEVANCE    = "relevance"     # reranker signal
    CONFIDENCE   = "confidence"    # classifier posterior
    QUALITY      = "quality"       # LLM-as-judge or heuristic
    SENTIMENT    = "sentiment"     # positive pole of a sentiment scale


@dataclass(slots=True, frozen=True)
class Score:
    value: float
    kind:  ScoreKind = ScoreKind.SIMILARITY
    low:   float     = 0.0
    high:  float     = 1.0

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"Score bounds invalid: low={self.low} >= high={self.high}")
        if not (self.low <= self.value <= self.high):
            raise ValueError(
                f"{self.kind.value} score {self.value} outside [{self.low}, {self.high}]"
            )

    # --- factories ---

    @classmethod
    def similarity(cls, value: float, *, low: float = 0.0, high: float = 1.0) -> Score:
        return cls(value, ScoreKind.SIMILARITY, low=low, high=high)

    @classmethod
    def relevance(cls, value: float) -> Score:
        return cls(value, ScoreKind.RELEVANCE)

    @classmethod
    def confidence(cls, value: float) -> Score:
        return cls(value, ScoreKind.CONFIDENCE)

    @classmethod
    def quality(cls, value: float) -> Score:
        return cls(value, ScoreKind.QUALITY)

    @classmethod
    def logit(cls, value: float, kind: ScoreKind = ScoreKind.CONFIDENCE) -> Score:
        return cls(value, kind, low=float("-inf"), high=float("inf"))

    # --- normalization ---

    @property
    def normalized(self) -> float:
        span = self.high - self.low
        if span == 0 or span == float("inf"):
            return self.value
        return (self.value - self.low) / span

    def to_percentage(self) -> float:
        return round(self.normalized * 100, 2)

    # --- comparison ---

    def __lt__(self, other: Score) -> bool:
        self._check_comparable(other)
        return self.value < other.value

    def __le__(self, other: Score) -> bool:
        self._check_comparable(other)
        return self.value <= other.value

    def __gt__(self, other: Score) -> bool:
        self._check_comparable(other)
        return self.value > other.value

    def __ge__(self, other: Score) -> bool:
        self._check_comparable(other)
        return self.value >= other.value

    def _check_comparable(self, other: object) -> None:
        if not isinstance(other, Score):
            raise TypeError(f"Cannot compare Score with {type(other)}")
        if self.kind != other.kind:
            raise TypeError(
                f"Cannot compare scores of different kinds: "
                f"{self.kind.value} vs {other.kind.value}"
            )

    # --- thresholds ---

    def exceeds(self, threshold: float) -> bool:
        return self.value > threshold

    def is_high_confidence(self, threshold: float = 0.85) -> bool:
        return self.normalized > threshold

    # --- display ---

    def __repr__(self) -> str:
        return f"Score({self.kind.value}={self.value:.4f}, range=[{self.low}, {self.high}])"