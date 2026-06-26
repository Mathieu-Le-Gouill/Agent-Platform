from typing import Protocol, runtime_checkable

from models.score import Score, ScoreKind


@runtime_checkable
class Scoreable(Protocol):
    @property
    def scores(self) -> dict[ScoreKind, Score]: ...