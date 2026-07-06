import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from pydantic import SecretStr

from agent_platform.integrations.vad.configuration import PvcobraVadConfig
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.span import SampleSpan
from agent_platform.models.enums import DataType


def test_config_defaults():
    cfg = PvcobraVadConfig()
    assert cfg.sample_rate == 16000
    assert cfg.threshold == 0.5


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
def test_provider_requirements(mock_pvcobra):
    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    mock_pvcobra.create.return_value = MagicMock()
    vad = PvcobraVAD(SecretStr("mock_access_key"))
    req = vad.requirements
    assert req.sample_rates == (16000,)
    assert req.channels == 1
    assert req.dtype == DataType.INT16


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
def test_requirements_rejects_invalid_sample_rate(mock_pvcobra):
    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    mock_pvcobra.create.return_value = MagicMock()
    vad = PvcobraVAD(SecretStr("mock_access_key"))
    chunk = AudioChunk(
        id=uuid4(),
        data=bytes(),
        sample_rate=10000,
        start=0,
        end=100,
        channels=1,
        dtype=DataType.INT16,
    )
    with pytest.raises(ValueError, match="Unsupported sample rate"):
        vad.requirements.validate(chunk)


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
def test_requirements_rejects_invalid_dtype(mock_pvcobra):
    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    mock_pvcobra.create.return_value = MagicMock()
    vad = PvcobraVAD(SecretStr("mock_access_key"))
    chunk = AudioChunk(
        id=uuid4(),
        data=bytes(),
        sample_rate=16000,
        start=0,
        end=100,
        channels=1,
        dtype=DataType.FLOAT32,
    )
    with pytest.raises(TypeError):
        vad.requirements.validate(chunk)


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
def test_requirements_rejects_invalid_channels(mock_pvcobra):
    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    mock_pvcobra.create.return_value = MagicMock()
    vad = PvcobraVAD(SecretStr("mock_access_key"))
    chunk = AudioChunk(
        id=uuid4(),
        data=bytes(),
        sample_rate=16000,
        start=0,
        end=100,
        channels=3,
        dtype=DataType.INT16,
    )
    with pytest.raises(ValueError, match="Expected 1 channel"):
        vad.requirements.validate(chunk)


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
def test_requirements_accepts_valid(mock_pvcobra):
    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    mock_pvcobra.create.return_value = MagicMock()
    vad = PvcobraVAD(SecretStr("mock_access_key"))
    chunk = AudioChunk(
        id=uuid4(),
        data=bytes(),
        sample_rate=16000,
        start=0,
        end=100,
        channels=1,
        dtype=DataType.INT16,
    )
    vad.requirements.validate(chunk)


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
def test_detect_with_speech_spans(mock_pvcobra):
    mock_handle = MagicMock()
    mock_handle.process.side_effect = [0.9, 0.8, 0.1]
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    vad = PvcobraVAD(SecretStr("mock_access_key"))

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


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
def test_is_speech_above_threshold(mock_pvcobra):
    mock_handle = MagicMock()
    mock_handle.process.return_value = 0.9
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    vad = PvcobraVAD(SecretStr("mock_access_key"))

    chunk = AudioChunk(
        id=uuid4(),
        data=b"\x00\x00" * 160,
        sample_rate=16000,
        start=0,
        end=160,
        channels=1,
        dtype=DataType.INT16,
    )
    config = PvcobraVadConfig()
    assert vad._is_speech(chunk, config) is True
    mock_handle.process.assert_called_once_with(chunk.data)


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
def test_is_speech_below_threshold(mock_pvcobra):
    mock_handle = MagicMock()
    mock_handle.process.return_value = 0.1
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    vad = PvcobraVAD(SecretStr("mock_access_key"))

    chunk = AudioChunk(
        id=uuid4(),
        data=b"\x00\x00" * 160,
        sample_rate=16000,
        start=0,
        end=160,
        channels=1,
        dtype=DataType.INT16,
    )
    config = PvcobraVadConfig()
    assert vad._is_speech(chunk, config) is False


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
def test_detect_empty_returns_empty(mock_pvcobra):
    mock_pvcobra.create.return_value = MagicMock()
    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    vad = PvcobraVAD(SecretStr("mock_access_key"))
    assert vad.detect([]) == []


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
def test_detect_no_span_when_silence_before_speech(mock_pvcobra):
    mock_handle = MagicMock()
    mock_handle.process.side_effect = [0.1, 0.1]
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    vad = PvcobraVAD(SecretStr("mock_access_key"))

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
    ]

    result = vad.detect(chunks)
    assert result == []


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
async def test_adetect_yields_span(mock_pvcobra):
    mock_handle = MagicMock()
    mock_handle.process.side_effect = [0.9, 0.1]
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    vad = PvcobraVAD(SecretStr("mock_access_key"))

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


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
async def test_adetect_validate_chunk_raises(mock_pvcobra):
    mock_handle = MagicMock()
    mock_handle.process.return_value = 0.1
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD

    vad = PvcobraVAD(SecretStr("mock_access_key"))

    async def _gen():
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00" * 300,
            sample_rate=8000,
            start=0,
            end=300,
            channels=1,
            dtype=DataType.INT16,
        )

    with pytest.raises((ValueError,), match="Unsupported sample rate"):
        async for _ in vad.adetect(_gen()):
            pass
