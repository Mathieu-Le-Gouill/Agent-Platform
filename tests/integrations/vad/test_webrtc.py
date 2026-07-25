from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("webrtcvad")

from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import DataType
from agent_platform.core.schemas.span import SampleSpan
from agent_platform.integrations.vad.webrtc.config import WebrtcVadConfig


def _chunk(num_samples: int, start: int, sample_rate: int = 16000) -> AudioChunk:
    return AudioChunk(
        id=uuid4(),
        data=b"\x00\x00" * num_samples,
        sample_rate=sample_rate,
        start=start,
        end=start + num_samples,
        channels=1,
        dtype=DataType.INT16,
    )


def test_config_defaults():
    cfg = WebrtcVadConfig()
    assert cfg.sample_rate == 16000
    assert cfg.mode == 1
    assert cfg.speech_pad_ms == 30
    assert cfg.min_silence_duration_ms == 100
    assert cfg.min_speech_duration_ms == 250


def test_aggressiveness_field_removed():
    assert "aggressiveness" not in WebrtcVadConfig.model_fields


def test_requirements(mocker):
    mocker.patch("agent_platform.integrations.vad.webrtc.provider.webrtcvad")
    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()
    req = vad.requirements
    assert req.sample_rates == (8000, 16000, 32000, 48000)
    assert req.channels == 1
    assert req.dtype == DataType.INT16
    assert req.normalized is False


def test_empty_sequence_detect(mocker):
    mocker.patch("agent_platform.integrations.vad.webrtc.provider.webrtcvad")
    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()
    assert vad.detect([]) == []


def test_constructor(mocker):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_vad = MagicMock()
    mock_webrtcvad.Vad.return_value = mock_vad
    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()
    assert vad.model is mock_vad
    mock_webrtcvad.Vad.assert_called_once_with()


def test_detect_with_speech_spans(mocker):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_vad = MagicMock()
    mock_vad.is_speech.side_effect = [True, True, False]
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()

    chunks = [
        _chunk(480, 0),
        _chunk(480, 480),
        _chunk(160, 960),
    ]

    result = vad.detect(chunks)
    assert len(result) == 1
    assert result[0] == SampleSpan(start=0, end=990)
    mock_vad.set_mode.assert_called_once_with(1)


def test_detect_sample_rate_mismatch_raises_value_error(mocker):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_vad = MagicMock()
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()

    chunks = [_chunk(160, 0, sample_rate=8000)]

    with pytest.raises(ValueError, match="configuration sample rate differ"):
        vad.detect(chunks)


def test_detect_invalid_frame_duration_raises_value_error(mocker):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_vad = MagicMock()
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()

    chunks = [_chunk(300, 0)]

    with pytest.raises(ValueError, match="requires 10/20/30ms frames"):
        vad.detect(chunks)


@pytest.mark.parametrize("num_samples", [160, 320, 480])
def test_validate_chunk_accepts_10_20_30ms_frames(mocker, num_samples):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_webrtcvad.Vad.return_value = MagicMock()
    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()
    chunk = _chunk(num_samples, 0)
    vad._validate_chunk(chunk, 16000)


def test_is_speech(mocker):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = True
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()

    chunk = _chunk(160, 0)
    config = WebrtcVadConfig()
    assert vad._is_speech(chunk, config) is True
    mock_vad.is_speech.assert_called_once_with(chunk.data, config.sample_rate)


def test_is_speech_passes_raw_bytes_not_numpy_array(mocker):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = True
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()

    chunk = _chunk(160, 0)
    vad._is_speech(chunk, WebrtcVadConfig())

    sent = mock_vad.is_speech.call_args.args[0]
    assert isinstance(sent, bytes)
    assert sent == chunk.data


def test_detect_no_span_when_silence_before_speech(mocker):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = False
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()

    chunks = [_chunk(480, 0)]

    result = vad.detect(chunks)
    assert result == []


async def test_adetect_yields_span(mocker):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_vad = MagicMock()
    mock_vad.is_speech.side_effect = [True, False]
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()

    async def _gen():
        yield _chunk(480, 0)
        yield _chunk(160, 480)

    results = [span async for span in vad.adetect(_gen())]
    assert len(results) == 1
    assert results[0] == SampleSpan(start=0, end=510)


async def test_adetect_validate_chunk_raises(mocker):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = True
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()

    async def _gen():
        yield _chunk(480, 0, sample_rate=8000)

    with pytest.raises(ValueError, match="configuration sample rate differ"):
        async for _ in vad.adetect(_gen()):
            pass


async def test_adetect_invalid_frame_duration_raises(mocker):
    mock_webrtcvad = mocker.patch(
        "agent_platform.integrations.vad.webrtc.provider.webrtcvad"
    )
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = True
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.provider import Webrtcvad

    vad = Webrtcvad()

    async def _gen():
        yield _chunk(300, 0)

    with pytest.raises(ValueError, match="requires 10/20/30ms frames"):
        async for _ in vad.adetect(_gen()):
            pass
