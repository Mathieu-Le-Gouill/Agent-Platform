from collections.abc import AsyncIterator, Callable, Iterator

import pytest

from agent_platform.core.interfaces.dataset.base import (
    BaseDatasetProvider,
    BaseDatasetSplit,
)
from agent_platform.core.interfaces.dataset.config import DatasetConfig
from agent_platform.core.schemas.enums import DatasetSplit


class _ConcreteSplit(BaseDatasetSplit[dict]):
    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def __iter__(self) -> Iterator[dict]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def map(self, fn: Callable[[dict], dict]) -> "_ConcreteSplit":
        return _ConcreteSplit([fn(r) for r in self._records])

    def filter(self, predicate: Callable[[dict], bool]) -> "_ConcreteSplit":
        return _ConcreteSplit([r for r in self._records if predicate(r)])

    def batch(self, batch_size: int, drop_last: bool = False) -> Iterator[list[dict]]:
        for i in range(0, len(self._records), batch_size):
            chunk = self._records[i : i + batch_size]
            if drop_last and len(chunk) < batch_size:
                continue
            yield chunk

    async def abatch(
        self, batch_size: int, drop_last: bool = False
    ) -> AsyncIterator[list[dict]]:
        for chunk in self.batch(batch_size, drop_last):
            yield chunk


class _ConcreteProvider(BaseDatasetProvider[dict, DatasetConfig]):
    def load(self, record_type, config=None):
        return {DatasetSplit.TRAIN: _ConcreteSplit([{"a": 1}, {"a": 2}])}

    async def aload(self, record_type, config=None):
        return self.load(record_type, config)


class TestBaseDatasetSplit:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseDatasetSplit()

    def test_concrete_subclass_iterates(self):
        split = _ConcreteSplit([{"a": 1}, {"a": 2}])
        assert list(split) == [{"a": 1}, {"a": 2}]
        assert len(split) == 2

    def test_map_and_filter(self):
        split = _ConcreteSplit([{"a": 1}, {"a": 2}, {"a": 3}])
        mapped = split.map(lambda r: {"a": r["a"] * 10})
        assert list(mapped) == [{"a": 10}, {"a": 20}, {"a": 30}]
        filtered = mapped.filter(lambda r: r["a"] > 15)
        assert list(filtered) == [{"a": 20}, {"a": 30}]

    def test_batch(self):
        split = _ConcreteSplit([{"a": i} for i in range(5)])
        batches = list(split.batch(2))
        assert batches == [
            [{"a": 0}, {"a": 1}],
            [{"a": 2}, {"a": 3}],
            [{"a": 4}],
        ]

    async def test_abatch(self):
        split = _ConcreteSplit([{"a": i} for i in range(3)])
        batches = [b async for b in split.abatch(2)]
        assert batches == [[{"a": 0}, {"a": 1}], [{"a": 2}]]


class TestBaseDatasetProvider:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseDatasetProvider()

    def test_concrete_subclass_can_be_instantiated(self):
        provider = _ConcreteProvider()
        assert isinstance(provider, BaseDatasetProvider)

    async def test_aload_delegates_to_load(self):
        provider = _ConcreteProvider()
        result = await provider.aload(dict)
        assert DatasetSplit.TRAIN in result
        assert list(result[DatasetSplit.TRAIN]) == [{"a": 1}, {"a": 2}]
