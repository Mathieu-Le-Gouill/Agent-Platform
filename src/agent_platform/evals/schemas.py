from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agent_platform.core.schemas.score import Score


class EvalCase(BaseModel, frozen=True):
    id: str
    input: str
    expected: str | None = None
    reference: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    threshold: float = 0.7


class EvalOutput(BaseModel, frozen=True):
    text: str
    tool_calls: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class EvalResult(BaseModel, frozen=True):
    case_id: str
    output: EvalOutput | None
    scores: dict[str, Score] = Field(default_factory=dict)
    passed: bool
    error: str | None = None
    tags: list[str] = Field(default_factory=list)


class EvalReport(BaseModel, frozen=True):
    run_id: UUID = Field(default_factory=uuid4)
    results: list[EvalResult] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def mean_score(self, scorer_name: str) -> float:
        values = [
            r.scores[scorer_name].normalized
            for r in self.results
            if scorer_name in r.scores
        ]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def by_tag(self) -> dict[str, float]:
        tag_results: dict[str, list[bool]] = {}
        for result in self.results:
            for tag in result.tags:
                tag_results.setdefault(tag, []).append(result.passed)
        return {tag: sum(passed) / len(passed) for tag, passed in tag_results.items()}


__all__ = ["EvalCase", "EvalOutput", "EvalResult", "EvalReport"]
