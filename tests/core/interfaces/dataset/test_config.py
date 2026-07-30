import pytest
from pydantic import ValidationError

from agent_platform.core.interfaces.dataset.config import DatasetConfig
from agent_platform.core.schemas.enums import DatasetSplit


def test_requires_path():
    with pytest.raises(ValidationError):
        DatasetConfig()


def test_defaults():
    config = DatasetConfig(path="json")
    assert config.name is None
    assert config.data_files is None
    assert config.split_ratios is None
    assert config.streaming is False
    assert config.seed == 42


def test_construction_with_split_ratios():
    config = DatasetConfig(
        path="imdb",
        split_ratios={DatasetSplit.TRAIN: 0.8, DatasetSplit.TEST: 0.2},
        streaming=True,
        seed=7,
    )
    assert config.path == "imdb"
    assert config.split_ratios == {DatasetSplit.TRAIN: 0.8, DatasetSplit.TEST: 0.2}
    assert config.streaming is True
    assert config.seed == 7
