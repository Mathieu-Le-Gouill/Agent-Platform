import pytest
from uuid import uuid4

from agent_platform.integrations.vad.base import BaseVAD
from agent_platform.integrations.vad.configuration import VADConfig
from agent_platform.integrations.vad.requirements import AudioRequirements
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.span import SampleSpan
from agent_platform.models.enums import DataType


class _ConcreteVAD(BaseVAD[VADConfig]):
    @property
    def requirements(self) -> AudioRequirements:
        return AudioRequirements(
            sample_rates=(8000, 16000),
            channels=1,
            dtype=DataType.FLOAT32,
            normalized=True,
        )

    def detect(self, audio_sequence, config=None) -> list[SampleSpan]:
        return []

    async def adetect(self, audio_sequence, config=None):
        return
        yield

    def _is_speech(self, chunk, config) -> bool:
        return False


def test_validate_chunk_matching_sample_rate():
    vad = _ConcreteVAD()
    chunk = AudioChunk(
        id=uuid4(),
        data=b"\x00\x00\x00\x00" * 160,
        sample_rate=16000,
        start=0,
        end=160,
        channels=1,
        dtype=DataType.FLOAT32,
    )
    vad._validate_chunk(chunk, 16000)


def test_validate_chunk_non_matching_sample_rate_raises():
    vad = _ConcreteVAD()
    chunk = AudioChunk(
        id=uuid4(),
        data=b"\x00\x00\x00\x00" * 160,
        sample_rate=8000,
        start=0,
        end=160,
        channels=1,
        dtype=DataType.FLOAT32,
    )
    with pytest.raises(ValueError, match="configuration sample rate differ"):
        vad._validate_chunk(chunk, 16000)


def test_validate_chunk_rejects_unsupported_sample_rate():
    vad = _ConcreteVAD()
    chunk = AudioChunk(
        id=uuid4(),
        data=b"\x00\x00\x00\x00" * 160,
        sample_rate=44100,
        start=0,
        end=160,
        channels=1,
        dtype=DataType.FLOAT32,
    )
    with pytest.raises(ValueError, match="Unsupported sample rate"):
        vad._validate_chunk(chunk, 16000)
