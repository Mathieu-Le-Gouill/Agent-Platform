from __future__ import annotations

from agent_platform.core.interfaces.dataset.config import DatasetConfig


class HuggingFaceDatasetConfig(DatasetConfig):
    revision: str | None = None
