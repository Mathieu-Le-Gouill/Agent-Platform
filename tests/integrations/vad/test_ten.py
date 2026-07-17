import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

pytest.importorskip("ten_vad")

from agent_platform.integrations.vad.ten.config import TenVadConfig
from agent_platform.core.interfaces.vad.requirements import AudioRequirements
from agent_platform.core.schemas.enums import DataType
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.span import SampleSpan


def test_config_defaults():
    cfg = TenVadConfig()
    assert cfg.sample_rate == 16000
    assert cfg.threshold == 0.5


def test_provider_requirements():
    req = AudioRequirements(
        sample_rates=(16000,),
        channels=1,
        dtype=DataType.INT16,
        normalized=True,
    )
    assert req.sample_rates == (16000,)
    assert req.channels == 1
    assert req.dtype == DataType.INT16


@patch("agent_platform.integrations.vad.ten.ten.TenVad")
def test_detect_with_speech(mock_ten_vad_cls):
    mock_handle = MagicMock()
    mock_handle.process.side_effect = [(0.9, 1), (0.8, 1), (0.1, 0)]
    mock_ten_vad_cls.return_value = mock_handle

    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()

    chunks = [
        AudioChunk(
            id=uuid4(),
            data=b"\x00\x00" * 300,
            sample_rate=16000,
            start=0,
            end=300,
            channels=1,
            dtype=DataType.INT16,
        ),
        AudioChunk(
            id=uuid4(),
            data=b"\x00\x00" * 300,
            sample_rate=16000,
            start=300,
            end=600,
            channels=1,
            dtype=DataType.INT16,
        ),
        AudioChunk(
            id=uuid4(),
            data=b"\x00\x00" * 150,
            sample_rate=16000,
            start=600,
            end=750,
            channels=1,
            dtype=DataType.INT16,
        ),
    ]

    result = vad.detect(chunks)
    assert len(result) == 1
    assert result[0] == SampleSpan(start=0, end=630)


@patch("agent_platform.integrations.vad.ten.ten.TenVad")
def test_is_speech_above_threshold(mock_ten_vad_cls):
    mock_handle = MagicMock()
    mock_handle.process.return_value = (0.9, 1)
    mock_ten_vad_cls.return_value = mock_handle

    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()
    vad.handle = mock_handle

    chunk = AudioChunk(
        id=uuid4(),
        data=b"\x00\x00" * 160,
        sample_rate=16000,
        start=0,
        end=160,
        channels=1,
        dtype=DataType.INT16,
    )
    config = TenVadConfig()
    assert vad._is_speech(chunk, config) is True


@patch("agent_platform.integrations.vad.ten.ten.TenVad")
def test_is_speech_below_threshold(mock_ten_vad_cls):
    mock_handle = MagicMock()
    mock_handle.process.return_value = (0.1, 0)
    mock_ten_vad_cls.return_value = mock_handle

    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()
    vad.handle = mock_handle

    chunk = AudioChunk(
        id=uuid4(),
        data=b"\x00\x00" * 160,
        sample_rate=16000,
        start=0,
        end=160,
        channels=1,
        dtype=DataType.INT16,
    )
    config = TenVadConfig()
    assert vad._is_speech(chunk, config) is False


@patch("agent_platform.integrations.vad.ten.ten.TenVad")
async def test_adetect_async_flow(mock_ten_vad_cls):
    mock_handle = MagicMock()
    mock_handle.process.side_effect = [(0.9, 1), (0.1, 0)]
    mock_ten_vad_cls.return_value = mock_handle

    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()
    vad.handle = mock_handle

    async def _gen():
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00\x00" * 300,
            sample_rate=16000,
            start=0,
            end=300,
            channels=1,
            dtype=DataType.INT16,
        )
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00\x00" * 150,
            sample_rate=16000,
            start=300,
            end=450,
            channels=1,
            dtype=DataType.INT16,
        )

    results = [span async for span in vad.adetect(_gen())]
    assert len(results) == 1
    assert results[0] == SampleSpan(start=0, end=330)


@patch("agent_platform.integrations.vad.ten.ten.TenVad")
def test_detect_empty_returns_empty(mock_ten_vad_cls):
    mock_ten_vad_cls.return_value = MagicMock()
    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()
    assert vad.detect([]) == []


@patch("agent_platform.integrations.vad.ten.ten.TenVad")
def test_detect_all_silence_returns_empty(mock_ten_vad_cls):
    mock_handle = MagicMock()
    mock_handle.process.return_value = (0.1, 0)
    mock_ten_vad_cls.return_value = mock_handle

    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()
    vad.handle = mock_handle

    chunks = [
        AudioChunk(
            id=uuid4(),
            data=b"\x00\x00" * 300,
            sample_rate=16000,
            start=0,
            end=300,
            channels=1,
            dtype=DataType.INT16,
        ),
    ]

    result = vad.detect(chunks)
    assert result == []
