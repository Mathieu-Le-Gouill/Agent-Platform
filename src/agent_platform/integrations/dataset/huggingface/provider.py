from __future__ import annotations

import asyncio
from typing import TypeVar

from datasets import load_dataset

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ConfigError, ProviderError
from agent_platform.core.interfaces.dataset.base import (
    BaseDatasetProvider,
    BaseDatasetSplit,
)
from agent_platform.core.schemas.enums import DatasetSplit
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.integrations.dataset.huggingface.config import (
    HuggingFaceDatasetConfig,
)
from agent_platform.integrations.dataset.huggingface.mappers import resolve_splits

RecordT = TypeVar("RecordT")


class HuggingFaceDatasetProvider(
    BaseDatasetProvider[RecordT, HuggingFaceDatasetConfig]
):
    def __init__(self, credentials: HuggingFaceCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, HuggingFaceCredentials)

    def load(
        self,
        record_type: type[RecordT],
        config: HuggingFaceDatasetConfig | None = None,
    ) -> dict[DatasetSplit, BaseDatasetSplit[RecordT]]:
        if config is None:
            raise ConfigError(
                "HuggingFaceDatasetProvider requires a config with `path` set"
            )

        token = (
            self._credentials.api_key.get_secret_value()
            if self._credentials.api_key is not None
            else None
        )
        try:
            raw = load_dataset(
                config.path,
                config.name,
                data_files=config.data_files,
                revision=config.revision,
                streaming=config.streaming,
                token=token,
            )
        except Exception as exc:
            raise ProviderError(
                f"failed to load dataset {config.path!r}: {exc}"
            ) from exc

        return resolve_splits(raw, record_type, config.split_ratios, config.seed)

    async def aload(
        self,
        record_type: type[RecordT],
        config: HuggingFaceDatasetConfig | None = None,
    ) -> dict[DatasetSplit, BaseDatasetSplit[RecordT]]:
        return await asyncio.to_thread(self.load, record_type, config)
