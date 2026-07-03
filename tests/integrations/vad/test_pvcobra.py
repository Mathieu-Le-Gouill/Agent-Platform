import pytest
from agent_platform.integrations.vad.providers.pvcobra import (
    PvcobraVAD,
    PvcobraVadConfig,
    AudioRequirements,
)
from agent_platform.utils.uuid import new_uuid
from pydantic import SecretStr
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.enums.dtype import DataType


def test_requirements():
    mock_requirements_validate(10000, DataType.INT16, 1) # Invalid sample_rate (10000 != 16000) -> raise value Error
    mock_requirements_validate(16000, DataType.FLOAT32, 1) # Invalid dtype -> raise value Error
    mock_requirements_validate(16000, DataType.INT16, 3) # Invalid channels nb -> raise value Error
    mock_requirements_validate(16000, DataType.INT16, 1) # Valid

def mock_requirements_validate(sample_rate: int, dtype: DataType, channels: int):
    vad = PvcobraVAD(SecretStr("mock_acess_key"))

    mock_chunk = AudioChunk(
            id=new_uuid(),
            data=bytes(),
            sample_rate=sample_rate,
            start=0,
            end=100,
            channels=channels,
            dtype=dtype
    )

    vad.requirements.validate(mock_chunk) 

    mock_chunk.sample_rate


def test_detect():
    ...

    
@pytest.mark.asyncio
def test_adetect():
    ...
    