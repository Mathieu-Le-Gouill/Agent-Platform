from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.safe_execution import safe_call
from agent_platform.core.interfaces.classification.base import (
    BaseClassificationProvider,
)
from agent_platform.core.interfaces.classification.response import (
    ClassificationResult,
)


class ClassifyInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")
    candidate_labels: list[str] = Field(
        ..., min_length=1, description="Candidate labels to classify the text into"
    )


class ClassifyTool(Tool):
    name = "classify"
    description = "Classify text into one of a set of candidate labels."
    input_schema = ClassifyInput
    output_schema = ClassificationResult

    def __init__(self, classifier: BaseClassificationProvider) -> None:
        self._classifier = classifier

    async def run(self, **kwargs: Any) -> ClassificationResult:
        validated = ClassifyInput(**kwargs)
        return await safe_call(
            self._classifier.classify(validated.text, validated.candidate_labels),
            "Classification failed",
        )
