from __future__ import annotations

from agent_platform.core.config import ProviderConfig
from agent_platform.core.schemas.enums import DatasetSplit


class DatasetConfig(ProviderConfig):
    # HF Hub dataset id (e.g. "imdb"), or a builder name ("json"/"csv"/"parquet")
    # when loading local files via `data_files`.
    path: str
    # HF dataset "config name", for multi-config hub datasets (e.g. glue's "mrpc").
    name: str | None = None
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    # Only used when the source has a single split (or none) to divide into
    # train/test/eval; ignored when the source already has native splits.
    split_ratios: dict[DatasetSplit, float] | None = None
    streaming: bool = False
    seed: int = 42
