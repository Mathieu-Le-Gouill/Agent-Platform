import pytest

pytest.importorskip("datasets")

import datasets
from pydantic import BaseModel

from agent_platform.core.errors import ConfigError
from agent_platform.core.schemas.enums import DatasetSplit
from agent_platform.integrations.dataset.huggingface.mappers import resolve_splits


class Record(BaseModel):
    a: int
    b: str


def _make_dataset(n: int) -> datasets.Dataset:
    return datasets.Dataset.from_list([{"a": i, "b": str(i)} for i in range(n)])


class TestResolveSplitsSingleDataset:
    def test_no_split_ratios_defaults_to_train(self):
        result = resolve_splits(_make_dataset(3), Record, None, seed=1)
        assert set(result) == {DatasetSplit.TRAIN}
        assert len(result[DatasetSplit.TRAIN]) == 3

    def test_split_ratios_train_test(self):
        result = resolve_splits(
            _make_dataset(10),
            Record,
            {DatasetSplit.TRAIN: 0.7, DatasetSplit.TEST: 0.3},
            seed=1,
        )
        assert set(result) == {DatasetSplit.TRAIN, DatasetSplit.TEST}
        assert len(result[DatasetSplit.TRAIN]) + len(result[DatasetSplit.TEST]) == 10

    def test_split_ratios_train_test_eval(self):
        result = resolve_splits(
            _make_dataset(20),
            Record,
            {DatasetSplit.TRAIN: 0.6, DatasetSplit.TEST: 0.2, DatasetSplit.EVAL: 0.2},
            seed=1,
        )
        assert set(result) == {DatasetSplit.TRAIN, DatasetSplit.TEST, DatasetSplit.EVAL}
        total = sum(len(split) for split in result.values())
        assert total == 20

    def test_split_ratios_must_be_positive(self):
        with pytest.raises(ConfigError):
            resolve_splits(
                _make_dataset(10),
                Record,
                {DatasetSplit.TRAIN: 0.0, DatasetSplit.TEST: 0.0},
                seed=1,
            )

    def test_split_ratios_unsupported_for_streaming(self):
        streaming = _make_dataset(10).to_iterable_dataset()
        with pytest.raises(ConfigError):
            resolve_splits(
                streaming,
                Record,
                {DatasetSplit.TRAIN: 0.5, DatasetSplit.TEST: 0.5},
                seed=1,
            )


class TestResolveSplitsDatasetDict:
    def test_recognized_split_names(self):
        dataset_dict = datasets.DatasetDict(
            {
                "train": _make_dataset(5),
                "validation": _make_dataset(2),
                "test": _make_dataset(1),
            }
        )
        result = resolve_splits(dataset_dict, Record, None, seed=1)
        assert set(result) == {DatasetSplit.TRAIN, DatasetSplit.EVAL, DatasetSplit.TEST}
        assert len(result[DatasetSplit.TRAIN]) == 5
        assert len(result[DatasetSplit.EVAL]) == 2
        assert len(result[DatasetSplit.TEST]) == 1

    def test_single_unrecognized_split_falls_back_to_train(self):
        dataset_dict = datasets.DatasetDict({"default": _make_dataset(4)})
        result = resolve_splits(dataset_dict, Record, None, seed=1)
        assert set(result) == {DatasetSplit.TRAIN}
        assert len(result[DatasetSplit.TRAIN]) == 4

    def test_multiple_unrecognized_splits_raise(self):
        dataset_dict = datasets.DatasetDict(
            {"foo": _make_dataset(2), "bar": _make_dataset(2)}
        )
        with pytest.raises(ConfigError):
            resolve_splits(dataset_dict, Record, None, seed=1)


class TestHFDatasetSplitAdapter:
    def test_iter_and_len(self):
        result = resolve_splits(_make_dataset(3), Record, None, seed=1)
        split = result[DatasetSplit.TRAIN]
        records = list(split)
        assert len(split) == 3
        assert all(isinstance(r, Record) for r in records)
        assert [r.a for r in records] == [0, 1, 2]

    def test_map(self):
        split = resolve_splits(_make_dataset(3), Record, None, seed=1)[
            DatasetSplit.TRAIN
        ]
        mapped = split.map(lambda r: Record(a=r.a * 10, b=r.b))
        assert [r.a for r in mapped] == [0, 10, 20]

    def test_filter(self):
        split = resolve_splits(_make_dataset(5), Record, None, seed=1)[
            DatasetSplit.TRAIN
        ]
        filtered = split.filter(lambda r: r.a >= 3)
        assert [r.a for r in filtered] == [3, 4]

    def test_batch(self):
        split = resolve_splits(_make_dataset(5), Record, None, seed=1)[
            DatasetSplit.TRAIN
        ]
        batches = list(split.batch(2))
        assert [len(b) for b in batches] == [2, 2, 1]
        assert all(isinstance(r, Record) for b in batches for r in b)

    async def test_abatch(self):
        split = resolve_splits(_make_dataset(3), Record, None, seed=1)[
            DatasetSplit.TRAIN
        ]
        batches = [b async for b in split.abatch(2)]
        assert [len(b) for b in batches] == [2, 1]
