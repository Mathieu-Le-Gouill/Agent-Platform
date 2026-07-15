from pydantic import BaseModel

from agent_platform.core.schemas.score import Score


class ClassificationPrediction(BaseModel):
    label: str
    score: Score


class ClassificationResult(BaseModel):
    predictions: list[ClassificationPrediction]

    @property
    def top(self) -> ClassificationPrediction | None:
        return self.predictions[0] if self.predictions else None

    @property
    def label(self) -> str | None:
        return self.top.label if self.top else None

    @property
    def score(self) -> Score | None:
        return self.top.score if self.top else None


class ClassificationResponse(BaseModel):
    results: list[ClassificationResult]
