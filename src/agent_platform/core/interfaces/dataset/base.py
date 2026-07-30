from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Generic, TypeVar

from agent_platform.core.interfaces.dataset.config import DatasetConfig
from agent_platform.core.schemas.enums import DatasetSplit

RecordT = TypeVar("RecordT")
ConfigT = TypeVar("ConfigT", bound=DatasetConfig, contravariant=True)


class BaseDatasetSplit(ABC, Generic[RecordT]):
    """One split (train/test/eval) of a dataset.

    Backed by whatever lazy/streaming source the provider loaded (e.g. an
    Arrow-backed `datasets.Dataset`); `map`/`filter` delegate to that source's
    own implementation rather than re-materializing records in Python, so
    caching/streaming behavior it already provides carries through.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[RecordT]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def map(self, fn: Callable[[RecordT], RecordT]) -> BaseDatasetSplit[RecordT]: ...

    @abstractmethod
    def filter(
        self, predicate: Callable[[RecordT], bool]
    ) -> BaseDatasetSplit[RecordT]: ...

    @abstractmethod
    def batch(
        self, batch_size: int, drop_last: bool = False
    ) -> Iterator[list[RecordT]]: ...

    @abstractmethod
    def abatch(
        self, batch_size: int, drop_last: bool = False
    ) -> AsyncIterator[list[RecordT]]: ...


class BaseDatasetProvider(ABC, Generic[RecordT, ConfigT]):
    @abstractmethod
    def load(
        self, record_type: type[RecordT], config: ConfigT | None = None
    ) -> dict[DatasetSplit, BaseDatasetSplit[RecordT]]: ...

    @abstractmethod
    async def aload(
        self, record_type: type[RecordT], config: ConfigT | None = None
    ) -> dict[DatasetSplit, BaseDatasetSplit[RecordT]]: ...
