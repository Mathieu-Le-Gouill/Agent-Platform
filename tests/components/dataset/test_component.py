from unittest.mock import AsyncMock

from pydantic import BaseModel

from agent_platform.components.dataset.component import Dataset
from agent_platform.core.interfaces.dataset.config import DatasetConfig
from agent_platform.core.schemas.enums import DatasetSplit


class Record(BaseModel):
    a: int


class TestDatasetComponent:
    async def test_arun_delegates_to_backend_aload_with_config(self):
        backend = AsyncMock()
        backend.aload.return_value = {DatasetSplit.TRAIN: "sentinel"}
        config = DatasetConfig(data_files="data.jsonl")
        component = Dataset(backend=backend, record_type=Record, config=config)

        result = await component.arun("json")

        backend.aload.assert_awaited_once_with(Record, "json", config)
        assert result == {DatasetSplit.TRAIN: "sentinel"}

    async def test_arun_defaults_config_to_none(self):
        backend = AsyncMock()
        backend.aload.return_value = {DatasetSplit.TRAIN: "sentinel"}
        component = Dataset(backend=backend, record_type=Record)

        result = await component.arun("json")

        backend.aload.assert_awaited_once_with(Record, "json", None)
        assert result == {DatasetSplit.TRAIN: "sentinel"}

    def test_run_is_sync_wrapper(self):
        backend = AsyncMock()
        backend.aload.return_value = {DatasetSplit.TRAIN: "sentinel"}
        component = Dataset(backend=backend, record_type=Record)

        result = component.run("json")

        assert result == {DatasetSplit.TRAIN: "sentinel"}
