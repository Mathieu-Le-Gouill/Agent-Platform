from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("silero_vad")

from agent_platform.core.interfaces.vad.requirements import AudioRequirements
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import DataType
from agent_platform.core.schemas.span import SampleSpan
from agent_platform.integrations.vad.silero.config import SileroVadConfig


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


def test_detect_with_non_empty_audio(mocker):
    mock_get_speech_timestamps = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.get_speech_timestamps"
    )
    mock_load_silero_vad = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.load_silero_vad"
    )
    mock_load_silero_vad.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [
        {"start": 0, "end": 16000},
    ]

    from agent_platform.integrations.vad.silero.provider import SileroVAD

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


def test_detect_produces_correct_sample_spans(mocker):
    mock_get_speech_timestamps = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.get_speech_timestamps"
    )
    mock_load_silero_vad = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.load_silero_vad"
    )
    mock_load_silero_vad.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [
        {"start": 100, "end": 500},
        {"start": 1000, "end": 2000},
    ]

    from agent_platform.integrations.vad.silero.provider import SileroVAD

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


async def test_adetect_async_flow(mocker):
    mock_get_speech_timestamps = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.get_speech_timestamps"
    )
    mock_load_silero_vad = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.load_silero_vad"
    )
    mock_load_silero_vad.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [
        {"start": 0, "end": 16000},
    ]

    from agent_platform.integrations.vad.silero.provider import SileroVAD

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


async def test_adetect_sample_rate_mismatch_raises_value_error(mocker):
    mock_load_silero_vad = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.load_silero_vad"
    )
    mock_load_silero_vad.return_value = MagicMock()

    from agent_platform.integrations.vad.silero.provider import SileroVAD

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


def test_detect_empty_returns_empty(mocker):
    mock_load = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.load_silero_vad"
    )
    mock_load.return_value = MagicMock()
    from agent_platform.integrations.vad.silero.provider import SileroVAD

    vad = SileroVAD()
    assert vad.detect([]) == []


def test_is_speech_raises_not_implemented(mocker):
    mock_load = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.load_silero_vad"
    )
    mock_load.return_value = MagicMock()
    from agent_platform.integrations.vad.silero.provider import SileroVAD

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


async def test_adetect_buffer_management(mocker):
    mock_get_speech_timestamps = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.get_speech_timestamps"
    )
    mock_load = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.load_silero_vad"
    )
    mock_load.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [{"start": 0, "end": 80}]

    from agent_platform.integrations.vad.silero.provider import SileroVAD

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


def test_config_default_max_samples_is_duration_appropriate():
    cfg = SileroVadConfig()
    assert cfg.max_samples == 80_000
    assert cfg.max_speech_duration_s == float("inf")


def test_detect_forwards_max_speech_duration_s(mocker):
    mock_get_speech_timestamps = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.get_speech_timestamps"
    )
    mock_load_silero_vad = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.load_silero_vad"
    )
    mock_load_silero_vad.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = []

    from agent_platform.integrations.vad.silero.provider import SileroVAD

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

    vad.detect(chunks, config=SileroVadConfig(max_speech_duration_s=15.0))

    _, kwargs = mock_get_speech_timestamps.call_args
    assert kwargs["max_speech_duration_s"] == 15.0


async def test_adetect_forwards_max_speech_duration_s(mocker):
    mock_get_speech_timestamps = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.get_speech_timestamps"
    )
    mock_load_silero_vad = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.load_silero_vad"
    )
    mock_load_silero_vad.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = []

    from agent_platform.integrations.vad.silero.provider import SileroVAD

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

    async for _ in vad.adetect(
        _gen(), config=SileroVadConfig(max_speech_duration_s=8.0)
    ):
        pass

    _, kwargs = mock_get_speech_timestamps.call_args
    assert kwargs["max_speech_duration_s"] == 8.0


async def test_adetect_buffer_trims_when_max_samples_exceeded(mocker):
    mock_get_speech_timestamps = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.get_speech_timestamps"
    )
    mock_load = mocker.patch(
        "agent_platform.integrations.vad.silero.provider.load_silero_vad"
    )
    mock_load.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [{"start": 0, "end": 10}]

    from agent_platform.integrations.vad.silero.provider import SileroVAD

    vad = SileroVAD()

    async def _gen():
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00\x00\x00\x00" * 40,
            sample_rate=16000,
            start=0,
            end=40,
            channels=1,
            dtype=DataType.FLOAT32,
        )
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00\x00\x00\x00" * 40,
            sample_rate=16000,
            start=40,
            end=80,
            channels=1,
            dtype=DataType.FLOAT32,
        )

    results = [
        span
        async for span in vad.adetect(_gen(), config=SileroVadConfig(max_samples=50))
    ]

    assert len(results) == 2
    last_call_audio = mock_get_speech_timestamps.call_args_list[-1][0][0]
    assert last_call_audio.shape[-1] <= 50
