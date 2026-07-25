from __future__ import annotations

from agent_platform.core.interfaces.classification.config import ClassificationConfig


class TransformersClassificationConfig(ClassificationConfig):
    model: str = "facebook/bart-large-mnli"
    device: str = "cpu"
    hypothesis_template: str = "This example is {}."
