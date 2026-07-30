from unittest.mock import AsyncMock

from pydantic import BaseModel

from agent_platform.components.dataset.component import Dataset
from agent_platform.core.interfaces.dataset.config import DatasetConfig
from agent_platform.core.schemas.enums import DatasetSplit


class Record(BaseModel):
    a: int


class TestDatasetComponent:
    async def test_arun_delegates_to_backend_aload(self):
        backend = AsyncMock()
        backend.aload.return_value = {DatasetSplit.TRAIN: "sentinel"}
        component = Dataset(backend=backend, record_type=Record)
        config = DatasetConfig(path="json", data_files="data.jsonl")

        result = await component.arun(config)

        backend.aload.assert_awaited_once_with(Record, config)
        assert result == {DatasetSplit.TRAIN: "sentinel"}

    def test_run_is_sync_wrapper(self):
        backend = AsyncMock()
        backend.aload.return_value = {DatasetSplit.TRAIN: "sentinel"}
        component = Dataset(backend=backend, record_type=Record)
        config = DatasetConfig(path="json", data_files="data.jsonl")

        result = component.run(config)

        assert result == {DatasetSplit.TRAIN: "sentinel"}
