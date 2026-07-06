import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from agent_platform.integrations.vad.configuration import WebrtcVadConfig
from agent_platform.models.enums import DataType
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.span import SampleSpan


def test_config_defaults():
    cfg = WebrtcVadConfig()
    assert cfg.sample_rate == 16000
    assert cfg.mode == 1
    assert cfg.speech_pad_ms == 30
    assert cfg.min_silence_duration_ms == 100
    assert cfg.min_speech_duration_ms == 250


def test_requirements():
    try:
        from agent_platform.integrations.vad.providers.webrtc import Webrtcvad
    except ImportError:
        pytest.skip("webrtcvad not installed")
    vad = Webrtcvad()
    req = vad.requirements
    assert req.sample_rates == (8000, 16000, 32000, 48000)
    assert req.channels == 1
    assert req.dtype == DataType.INT16
    assert req.normalized is False


def test_empty_sequence_detect():
    try:
        from agent_platform.integrations.vad.providers.webrtc import Webrtcvad
    except ImportError:
        pytest.skip("webrtcvad not installed")
    vad = Webrtcvad()
    result = vad.detect([])
    assert result == []


def test_provider_instantiation_requires_webrtcvad():
    try:
        from agent_platform.integrations.vad.providers.webrtc import Webrtcvad
    except ImportError as e:
        pytest.skip(f"webrtcvad dependency missing: {e}")


@patch("agent_platform.integrations.vad.providers.webrtc.webrtcvad")
def test_constructor(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_webrtcvad.Vad.return_value = mock_vad
    from agent_platform.integrations.vad.providers.webrtc import Webrtcvad

    vad = Webrtcvad()
    assert vad.model is mock_vad
    mock_webrtcvad.Vad.assert_called_once_with()


@patch("agent_platform.integrations.vad.providers.webrtc.webrtcvad")
def test_detect_with_speech_spans(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_vad.is_speech.side_effect = [True, True, False]
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.providers.webrtc import Webrtcvad

    vad = Webrtcvad()

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
    mock_vad.set_mode.assert_called_once_with(1)


@patch("agent_platform.integrations.vad.providers.webrtc.webrtcvad")
def test_detect_sample_rate_mismatch_raises_value_error(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.providers.webrtc import Webrtcvad

    vad = Webrtcvad()

    chunks = [
        AudioChunk(
            id=uuid4(),
            data=b"\x00\x00" * 160,
            sample_rate=8000,
            start=0,
            end=160,
            channels=1,
            dtype=DataType.INT16,
        ),
    ]

    with pytest.raises(ValueError, match="Expected 16000, got 8000"):
        vad.detect(chunks)


@patch("agent_platform.integrations.vad.providers.webrtc.webrtcvad")
def test_is_speech(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = True
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.providers.webrtc import Webrtcvad

    vad = Webrtcvad()

    chunk = AudioChunk(
        id=uuid4(),
        data=b"\x00\x00" * 160,
        sample_rate=16000,
        start=0,
        end=160,
        channels=1,
        dtype=DataType.INT16,
    )
    config = WebrtcVadConfig()
    assert vad._is_speech(chunk, config) is True
    mock_vad.is_speech.assert_called_once()


@patch("agent_platform.integrations.vad.providers.webrtc.webrtcvad")
def test_detect_no_span_when_silence_before_speech(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = False
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.providers.webrtc import Webrtcvad

    vad = Webrtcvad()

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


@patch("agent_platform.integrations.vad.providers.webrtc.webrtcvad")
async def test_adetect_yields_span(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_vad.is_speech.side_effect = [True, False]
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.providers.webrtc import Webrtcvad

    vad = Webrtcvad()

    async def _gen():
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00" * 300,
            sample_rate=16000,
            start=0,
            end=300,
            channels=1,
            dtype=DataType.INT16,
        )
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00" * 150,
            sample_rate=16000,
            start=300,
            end=450,
            channels=1,
            dtype=DataType.INT16,
        )

    results = [span async for span in vad.adetect(_gen())]
    assert len(results) == 1
    assert results[0] == SampleSpan(start=0, end=330)


@patch("agent_platform.integrations.vad.providers.webrtc.webrtcvad")
async def test_adetect_validate_chunk_raises(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = True
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.providers.webrtc import Webrtcvad

    vad = Webrtcvad()

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

    with pytest.raises(ValueError, match="configuration sample rate differ"):
        async for _ in vad.adetect(_gen()):
            pass
