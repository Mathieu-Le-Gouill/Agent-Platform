from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any, TypeAlias, TypeVar

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from agent_platform.core.errors import ConfigError
from agent_platform.core.interfaces.dataset.base import BaseDatasetSplit
from agent_platform.core.schemas.enums import DatasetSplit

__all__ = ["resolve_splits"]

RecordT = TypeVar("RecordT")

_HFDataset: TypeAlias = Dataset | IterableDataset
_HFDatasetDict: TypeAlias = DatasetDict | IterableDatasetDict

_SPLIT_ALIASES: dict[str, DatasetSplit] = {
    "train": DatasetSplit.TRAIN,
    "test": DatasetSplit.TEST,
    "validation": DatasetSplit.EVAL,
    "valid": DatasetSplit.EVAL,
    "val": DatasetSplit.EVAL,
    "dev": DatasetSplit.EVAL,
    "eval": DatasetSplit.EVAL,
}


class _HFDatasetSplit(BaseDatasetSplit[RecordT]):
    """Adapts a single Arrow-backed (or streaming) `datasets` split to `BaseDatasetSplit`.

    `map`/`filter`/`batch` delegate straight to the underlying `datasets.Dataset`
    methods, so its caching/fingerprinting and Arrow/streaming backing carry
    through unchanged; this class only converts rows to/from `record_type`.
    """

    def __init__(self, backend: _HFDataset, record_type: type[RecordT]) -> None:
        self._backend = backend
        self._record_type = record_type

    def __iter__(self) -> Iterator[RecordT]:
        for row in self._backend:
            yield self._record_type(**row)

    def __len__(self) -> int:
        return len(self._backend)

    def map(self, fn: Callable[[RecordT], RecordT]) -> _HFDatasetSplit[RecordT]:
        def _row_fn(row: dict[str, Any]) -> dict[str, Any]:
            record = fn(self._record_type(**row))
            return _model_dump(record)

        return _HFDatasetSplit(self._backend.map(_row_fn), self._record_type)

    def filter(self, predicate: Callable[[RecordT], bool]) -> _HFDatasetSplit[RecordT]:
        def _row_predicate(row: dict[str, Any]) -> bool:
            return predicate(self._record_type(**row))

        return _HFDatasetSplit(self._backend.filter(_row_predicate), self._record_type)

    def batch(
        self, batch_size: int, drop_last: bool = False
    ) -> Iterator[list[RecordT]]:
        for columnar in self._backend.iter(
            batch_size=batch_size, drop_last_batch=drop_last
        ):
            yield _rows_from_columnar(columnar, self._record_type)

    async def abatch(
        self, batch_size: int, drop_last: bool = False
    ) -> AsyncIterator[list[RecordT]]:
        for record_batch in self.batch(batch_size, drop_last):
            yield record_batch


def _model_dump(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        return record.model_dump()
    return dict(record)


def _rows_from_columnar(
    columnar: dict[str, list[Any]], record_type: type[RecordT]
) -> list[RecordT]:
    keys = list(columnar.keys())
    length = len(next(iter(columnar.values()), []))
    return [
        record_type(**{key: columnar[key][i] for key in keys}) for i in range(length)
    ]


def resolve_splits(
    dataset: _HFDataset | _HFDatasetDict,
    record_type: type[RecordT],
    split_ratios: dict[DatasetSplit, float] | None,
    seed: int,
) -> dict[DatasetSplit, BaseDatasetSplit[RecordT]]:
    if isinstance(dataset, DatasetDict | IterableDatasetDict):
        named = _match_named_splits(dataset)
        if named is not None:
            return {
                split: _HFDatasetSplit(hf_split, record_type)
                for split, hf_split in named.items()
            }
        if len(dataset) != 1:
            raise ConfigError(
                f"cannot map dataset splits {list(dataset.keys())} to train/test/eval; "
                "set split_ratios to divide a single split"
            )
        (dataset,) = dataset.values()

    if split_ratios:
        return _split_by_ratio(dataset, record_type, split_ratios, seed)

    return {DatasetSplit.TRAIN: _HFDatasetSplit(dataset, record_type)}


def _match_named_splits(
    dataset_dict: _HFDatasetDict,
) -> dict[DatasetSplit, _HFDataset] | None:
    resolved: dict[DatasetSplit, _HFDataset] = {}
    for name, hf_split in dataset_dict.items():
        split = _SPLIT_ALIASES.get(name.lower())
        if split is None:
            return None
        resolved[split] = hf_split
    return resolved


def _split_by_ratio(
    dataset: _HFDataset,
    record_type: type[RecordT],
    split_ratios: dict[DatasetSplit, float],
    seed: int,
) -> dict[DatasetSplit, BaseDatasetSplit[RecordT]]:
    if isinstance(dataset, IterableDataset):
        raise ConfigError("split_ratios is not supported with streaming=True")

    total = sum(split_ratios.values())
    if total <= 0:
        raise ConfigError("split_ratios must sum to a positive value")

    remaining = dataset
    result: dict[DatasetSplit, Dataset] = {}

    eval_ratio = split_ratios.get(DatasetSplit.EVAL, 0.0) / total
    if eval_ratio:
        split_out = remaining.train_test_split(test_size=eval_ratio, seed=seed)
        result[DatasetSplit.EVAL] = split_out["test"]
        remaining = split_out["train"]

    train_ratio = split_ratios.get(DatasetSplit.TRAIN, 0.0)
    test_ratio = split_ratios.get(DatasetSplit.TEST, 0.0)
    if test_ratio:
        denom = train_ratio + test_ratio
        if denom <= 0:
            raise ConfigError("split_ratios must sum to a positive value")
        split_out = remaining.train_test_split(test_size=test_ratio / denom, seed=seed)
        result[DatasetSplit.TEST] = split_out["test"]
        remaining = split_out["train"]

    result[DatasetSplit.TRAIN] = remaining
    return {
        split: _HFDatasetSplit(hf_split, record_type)
        for split, hf_split in result.items()
    }
