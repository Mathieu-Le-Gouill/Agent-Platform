import pytest

pytest.importorskip("datasets")

import datasets
from pydantic import BaseModel

from agent_platform.core.errors import ProviderError
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
    def test_load_uses_default_config_when_none_given(self, mocker):
        mock_load_dataset = mocker.patch(
            "agent_platform.integrations.dataset.huggingface.provider.load_dataset",
            return_value=datasets.Dataset.from_list([{"a": 1}, {"a": 2}]),
        )
        provider = HuggingFaceDatasetProvider(
            credentials=HuggingFaceCredentials(api_key=None)
        )

        result = provider.load(Record, "json")

        mock_load_dataset.assert_called_once_with(
            "json",
            None,
            data_files=None,
            revision=None,
            streaming=False,
            token=None,
        )
        assert set(result) == {DatasetSplit.TRAIN}
        assert [r.a for r in result[DatasetSplit.TRAIN]] == [1, 2]

    def test_load_calls_datasets_load_dataset(self, mocker):
        mock_load_dataset = mocker.patch(
            "agent_platform.integrations.dataset.huggingface.provider.load_dataset",
            return_value=datasets.Dataset.from_list([{"a": 1}, {"a": 2}]),
        )
        provider = HuggingFaceDatasetProvider(
            credentials=HuggingFaceCredentials(api_key=None)
        )
        config = HuggingFaceDatasetConfig(data_files="data.jsonl")

        result = provider.load(Record, "json", config)

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

        provider.load(Record, "some/dataset")

        assert mock_load_dataset.call_args.kwargs["token"] == "secret-token"

    def test_load_wraps_errors_in_provider_error(self, mocker):
        mocker.patch(
            "agent_platform.integrations.dataset.huggingface.provider.load_dataset",
            side_effect=RuntimeError("boom"),
        )
        provider = HuggingFaceDatasetProvider()

        with pytest.raises(ProviderError):
            provider.load(Record, "does/not-exist")

    async def test_aload_delegates_to_load(self, mocker):
        mocker.patch(
            "agent_platform.integrations.dataset.huggingface.provider.load_dataset",
            return_value=datasets.Dataset.from_list([{"a": 1}]),
        )
        provider = HuggingFaceDatasetProvider()
        config = HuggingFaceDatasetConfig(data_files="data.jsonl")

        result = await provider.aload(Record, "json", config)

        assert set(result) == {DatasetSplit.TRAIN}
        assert [r.a for r in result[DatasetSplit.TRAIN]] == [1]
