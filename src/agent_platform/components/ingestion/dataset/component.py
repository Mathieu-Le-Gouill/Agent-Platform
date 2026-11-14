from __future__ import annotations

from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.dataset.base import (
    BaseDatasetProvider,
    BaseDatasetSplit,
)
from agent_platform.core.interfaces.dataset.config import DatasetConfig
from agent_platform.core.schemas.enums import DatasetSplit

RecordT = TypeVar("RecordT")
DatasetConfigT = TypeVar("DatasetConfigT", bound=DatasetConfig)

DatasetInput = tuple[str, DatasetConfigT | None]


class Dataset(
    Component[
        DatasetInput[DatasetConfigT], dict[DatasetSplit, BaseDatasetSplit[RecordT]]
    ],
    Generic[RecordT, DatasetConfigT],
):
    def __init__(
        self,
        backend: BaseDatasetProvider[RecordT, DatasetConfigT],
        record_type: type[RecordT],
    ) -> None:
        self._backend = backend
        self._record_type = record_type

    async def arun(
        self, input: DatasetInput[DatasetConfigT]
    ) -> dict[DatasetSplit, BaseDatasetSplit[RecordT]]:
        path, config = input
        return await self._backend.aload(self._record_type, path, config)
