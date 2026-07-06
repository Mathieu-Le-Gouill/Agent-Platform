import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from agent_platform.integrations.vad.configuration import SileroVadConfig
from agent_platform.integrations.vad.requirements import AudioRequirements
from agent_platform.models.enums import DataType
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.span import SampleSpan


def test_config_defaults():
    cfg = SileroVadConfig()
    assert cfg.sample_rate == 16000
    assert cfg.threshold == 0.5


def test_requirements():
    req = AudioRequirements(
        sample_rates=(8000, 16000),
        channels=1,
        dtype=DataType.FLOAT32,
        normalized=True,
    )
    assert 16000 in req.sample_rates
    assert req.channels == 1
    assert req.dtype == DataType.FLOAT32


@patch("agent_platform.integrations.vad.providers.silero.load_silero_vad")
@patch("agent_platform.integrations.vad.providers.silero.get_speech_timestamps")
def test_detect_with_non_empty_audio(mock_get_speech_timestamps, mock_load_silero_vad):
    mock_load_silero_vad.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [
        {"start": 0, "end": 16000},
    ]

    from agent_platform.integrations.vad.providers.silero import SileroVAD

    vad = SileroVAD()
    chunks = [
        AudioChunk(
            id=uuid4(),
            data=b"\x00\x00\x00\x00" * 8000,
            sample_rate=16000,
            start=0,
            end=16000,
            channels=1,
            dtype=DataType.FLOAT32,
        ),
    ]
    spans = vad.detect(chunks)
    assert spans == [SampleSpan(start=0, end=16000)]


@patch("agent_platform.integrations.vad.providers.silero.load_silero_vad")
@patch("agent_platform.integrations.vad.providers.silero.get_speech_timestamps")
def test_detect_produces_correct_sample_spans(
    mock_get_speech_timestamps, mock_load_silero_vad
):
    mock_load_silero_vad.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [
        {"start": 100, "end": 500},
        {"start": 1000, "end": 2000},
    ]

    from agent_platform.integrations.vad.providers.silero import SileroVAD

    vad = SileroVAD()
    chunks = [
        AudioChunk(
            id=uuid4(),
            data=b"\x00\x00\x00\x00" * 8000,
            sample_rate=16000,
            start=0,
            end=16000,
            channels=1,
            dtype=DataType.FLOAT32,
        ),
    ]
    spans = vad.detect(chunks)
    assert spans == [
        SampleSpan(start=100, end=500),
        SampleSpan(start=1000, end=2000),
    ]


@patch("agent_platform.integrations.vad.providers.silero.load_silero_vad")
@patch("agent_platform.integrations.vad.providers.silero.get_speech_timestamps")
async def test_adetect_async_flow(mock_get_speech_timestamps, mock_load_silero_vad):
    mock_load_silero_vad.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [
        {"start": 0, "end": 16000},
    ]

    from agent_platform.integrations.vad.providers.silero import SileroVAD

    vad = SileroVAD()

    async def _gen():
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00\x00\x00\x00" * 40,
            sample_rate=16000,
            start=0,
            end=16000,
            channels=1,
            dtype=DataType.FLOAT32,
        )

    results = [span async for span in vad.adetect(_gen())]
    assert results == [SampleSpan(start=0, end=16000)]


@patch("agent_platform.integrations.vad.providers.silero.load_silero_vad")
async def test_adetect_sample_rate_mismatch_raises_value_error(mock_load_silero_vad):
    mock_load_silero_vad.return_value = MagicMock()

    from agent_platform.integrations.vad.providers.silero import SileroVAD

    vad = SileroVAD()

    async def _gen():
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00\x00\x00\x00" * 4000,
            sample_rate=8000,
            start=0,
            end=8000,
            channels=1,
            dtype=DataType.FLOAT32,
        )

    with pytest.raises(ValueError, match="Expected 16000, got 8000"):
        async for _ in vad.adetect(_gen()):
            pass


@patch("agent_platform.integrations.vad.providers.silero.load_silero_vad")
def test_detect_empty_returns_empty(mock_load):
    mock_load.return_value = MagicMock()
    from agent_platform.integrations.vad.providers.silero import SileroVAD

    vad = SileroVAD()
    assert vad.detect([]) == []


@patch("agent_platform.integrations.vad.providers.silero.load_silero_vad")
def test_is_speech_raises_not_implemented(mock_load):
    mock_load.return_value = MagicMock()
    from agent_platform.integrations.vad.providers.silero import SileroVAD

    vad = SileroVAD()
    chunk = AudioChunk(
        id=uuid4(),
        data=b"\x00\x00\x00\x00" * 40,
        sample_rate=16000,
        start=0,
        end=160,
        channels=1,
        dtype=DataType.FLOAT32,
    )
    with pytest.raises(NotImplementedError, match="SileroVAD uses batched detection"):
        vad._is_speech(chunk, SileroVadConfig())


@patch("agent_platform.integrations.vad.providers.silero.load_silero_vad")
@patch("agent_platform.integrations.vad.providers.silero.get_speech_timestamps")
async def test_adetect_buffer_management(mock_get_speech_timestamps, mock_load):
    mock_load.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [{"start": 0, "end": 80}]

    from agent_platform.integrations.vad.providers.silero import SileroVAD

    vad = SileroVAD()

    async def _gen():
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00\x00\x00\x00" * 40,
            sample_rate=16000,
            start=0,
            end=160,
            channels=1,
            dtype=DataType.FLOAT32,
        )
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00\x00\x00\x00" * 40,
            sample_rate=16000,
            start=160,
            end=320,
            channels=1,
            dtype=DataType.FLOAT32,
        )

    results = [span async for span in vad.adetect(_gen())]
    assert results == [SampleSpan(start=0, end=80), SampleSpan(start=0, end=80)]
