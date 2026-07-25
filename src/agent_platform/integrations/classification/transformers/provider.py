from __future__ import annotations

import asyncio
from typing import Any

from transformers import pipeline

from agent_platform.core.interfaces.classification.base import (
    BaseClassificationProvider,
)
from agent_platform.core.interfaces.classification.response import (
    ClassificationPrediction,
    ClassificationResult,
)
from agent_platform.core.schemas.score import Score, ScoreKind
from agent_platform.integrations.classification.transformers.config import (
    TransformersClassificationConfig,
)


class TransformersClassifier(
    BaseClassificationProvider[TransformersClassificationConfig]
):
    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}

    def _default_config(self) -> TransformersClassificationConfig:
        return TransformersClassificationConfig()

    def _load_pipeline_sync(self, config: TransformersClassificationConfig) -> Any:
        if config.model not in self._pipelines:
            self._pipelines[config.model] = pipeline(
                "zero-shot-classification",
                model=config.model,
                device=config.device,
            )
        return self._pipelines[config.model]

    async def _load_pipeline(self, config: TransformersClassificationConfig) -> Any:
        return await asyncio.to_thread(self._load_pipeline_sync, config)

    async def classify(
        self,
        text: str,
        candidate_labels: list[str],
        config: TransformersClassificationConfig | None = None,
    ) -> ClassificationResult:
        config = config or self._default_config()
        classifier = await self._load_pipeline(config)

        result = await asyncio.to_thread(
            classifier,
            text,
            candidate_labels,
            multi_label=config.multi_label,
            hypothesis_template=config.hypothesis_template,
        )

        predictions = [
            ClassificationPrediction(
                label=label, score=Score(value=score, kind=ScoreKind.CONFIDENCE)
            )
            for label, score in zip(result["labels"], result["scores"])
        ]
        return ClassificationResult(predictions=predictions)
