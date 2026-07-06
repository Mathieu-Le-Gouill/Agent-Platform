from __future__ import annotations

import asyncio
from typing import Sequence

from transformers import pipeline

from agent_platform.integrations.classification.base import ClassificationModel
from agent_platform.models.chunk import TextChunk


class ZeroShotClassifier(ClassificationModel):
    def __init__(
        self,
        model: str = "facebook/bart-large-mnli",
        device: int = -1,
        hypothesis_template: str = "This example is {}",
    ) -> None:

        self._model_name = model
        self._device = device
        self._hypothesis_template = hypothesis_template
        self._pipeline = None

    async def _load(self) -> None:
        if self._pipeline is not None:
            return

        loop = asyncio.get_event_loop()
        self._pipeline = await loop.run_in_executor(
            None,
            lambda: pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=self._device,
            ),
        )

    async def classify(
        self,
        items: list[TextChunk],
        candidate_labels: list[str] | None = None,
    ) -> list[str]:

        pipeline = self._pipeline

        if pipeline is None:
            raise RuntimeError("Failed to load Zero Shot Classification pipeline")

        if candidate_labels is None:
            labels = self._infer_labels(items)
        else:
            labels = candidate_labels

        await self._load()

        results: list[str] = []
        loop = asyncio.get_event_loop()

        for item in items:
            if not item.text.strip():
                results.append("")
                continue

            output = await loop.run_in_executor(
                None,
                lambda: pipeline(
                    item.text,
                    candidate_labels=labels,
                    hypothesis_template=self._hypothesis_template,
                ),
            )

            if output:
                results.append(str(output["labels"][0]))  # type: ignore[index]

        return results

    @staticmethod
    def _infer_labels(items: Sequence[TextChunk]) -> list[str]:

        seen: set[str] = set()
        labels: list[str] = []
        for item in items:
            meta_label = item.metadata.get("label")
            if meta_label and meta_label not in seen:
                seen.add(meta_label)
                labels.append(str(meta_label))
        return labels or ["positive", "negative", "neutral"]
