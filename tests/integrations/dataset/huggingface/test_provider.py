import pytest

pytest.importorskip("datasets")

import datasets
from pydantic import BaseModel

from agent_platform.core.errors import ConfigError, ProviderError
from agent_platform.core.schemas.enums import DatasetSplit
from agent_platform.integrations.credentials import HuggingFaceCredentials
from agent_platform.integrations.dataset.huggingface.config import (
    HuggingFaceDatasetConfig,
)
from agent_platform.integrations.dataset.huggingface.provider import (
    HuggingFaceDatasetProvider,
)


class Record(BaseModel):
    a: int


class TestHuggingFaceDatasetProvider:
    def test_load_requires_config(self):
        provider = HuggingFaceDatasetProvider()
        with pytest.raises(ConfigError):
            provider.load(Record, None)

    def test_load_calls_datasets_load_dataset(self, mocker):
        mock_load_dataset = mocker.patch(
            "agent_platform.integrations.dataset.huggingface.provider.load_dataset",
            return_value=datasets.Dataset.from_list([{"a": 1}, {"a": 2}]),
        )
        provider = HuggingFaceDatasetProvider(
            credentials=HuggingFaceCredentials(api_key=None)
        )
        config = HuggingFaceDatasetConfig(path="json", data_files="data.jsonl")

        result = provider.load(Record, config)

        mock_load_dataset.assert_called_once_with(
            "json",
            None,
            data_files="data.jsonl",
            revision=None,
            streaming=False,
            token=None,
        )
        assert set(result) == {DatasetSplit.TRAIN}
        assert [r.a for r in result[DatasetSplit.TRAIN]] == [1, 2]

    def test_load_passes_token_from_credentials(self, mocker):
        mock_load_dataset = mocker.patch(
            "agent_platform.integrations.dataset.huggingface.provider.load_dataset",
            return_value=datasets.Dataset.from_list([{"a": 1}]),
        )
        provider = HuggingFaceDatasetProvider(
            credentials=HuggingFaceCredentials(api_key="secret-token")
        )
        config = HuggingFaceDatasetConfig(path="some/dataset")

        provider.load(Record, config)

        assert mock_load_dataset.call_args.kwargs["token"] == "secret-token"

    def test_load_wraps_errors_in_provider_error(self, mocker):
        mocker.patch(
            "agent_platform.integrations.dataset.huggingface.provider.load_dataset",
            side_effect=RuntimeError("boom"),
        )
        provider = HuggingFaceDatasetProvider()
        config = HuggingFaceDatasetConfig(path="does/not-exist")

        with pytest.raises(ProviderError):
            provider.load(Record, config)

    async def test_aload_delegates_to_load(self, mocker):
        mocker.patch(
            "agent_platform.integrations.dataset.huggingface.provider.load_dataset",
            return_value=datasets.Dataset.from_list([{"a": 1}]),
        )
        provider = HuggingFaceDatasetProvider()
        config = HuggingFaceDatasetConfig(path="json", data_files="data.jsonl")

        result = await provider.aload(Record, config)

        assert set(result) == {DatasetSplit.TRAIN}
        assert [r.a for r in result[DatasetSplit.TRAIN]] == [1]
