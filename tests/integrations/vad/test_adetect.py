import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

pytest.importorskip("webrtcvad")

from agent_platform.integrations.vad.webrtc.config import WebrtcVadConfig
from agent_platform.integrations.vad.pvcobra.config import PvcobraVadConfig
from agent_platform.integrations.vad.silero.config import SileroVadConfig
from agent_platform.integrations.vad.ten.config import TenVadConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.span import SampleSpan
from agent_platform.core.schemas.enums import DataType

_HAS_TEN_VAD = True
try:
    __import__("ten_vad")
except ImportError:
    _HAS_TEN_VAD = False

_HAS_PVCOBRA = True
try:
    __import__("pvcobra")
except ImportError:
    _HAS_PVCOBRA = False


def _int16_chunk(num_samples: int, start: int, sample_rate: int = 16000) -> AudioChunk:
    return AudioChunk(
        id=uuid4(),
        data=b"\x00\x00" * num_samples,
        sample_rate=sample_rate,
        start=start,
        end=start + num_samples,
        channels=1,
        dtype=DataType.INT16,
    )


@patch("agent_platform.integrations.vad.webrtc.webrtc.webrtcvad")
async def test_webrtc_adetect_yields_span(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_vad.is_speech.side_effect = [True, True, False]
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.webrtc import Webrtcvad

    vad = Webrtcvad()

    async def _gen():
        yield _int16_chunk(480, 0)
        yield _int16_chunk(480, 480)
        yield _int16_chunk(160, 960)

    results = [span async for span in vad.adetect(_gen())]
    assert len(results) == 1
    assert results[0] == SampleSpan(start=0, end=990)


@patch("agent_platform.integrations.vad.webrtc.webrtc.webrtcvad")
async def test_webrtc_adetect_all_silence(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = False
    mock_webrtcvad.Vad.return_value = mock_vad

    from agent_platform.integrations.vad.webrtc.webrtc import Webrtcvad

    vad = Webrtcvad()

    async def _gen():
        yield _int16_chunk(480, 0)

    results = [span async for span in vad.adetect(_gen())]
    assert results == []


@pytest.mark.skipif(not _HAS_PVCOBRA, reason="pvcobra not installed")
@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
async def test_pvcobra_adetect_yields_span(mock_pvcobra):
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD
    from agent_platform.integrations.credentials import PicoVoiceCredentials
    from pydantic import SecretStr

    mock_handle = MagicMock()
    mock_handle.frame_length = 512
    mock_handle.process.side_effect = [0.9, 0.1]
    mock_pvcobra.create.return_value = mock_handle

    vad = PvcobraVAD(PicoVoiceCredentials(access_key=SecretStr("mock_access_key")))

    async def _gen():
        yield _int16_chunk(512, 0)
        yield _int16_chunk(512, 512)

    results = [span async for span in vad.adetect(_gen())]
    assert len(results) == 1


@pytest.mark.skipif(not _HAS_PVCOBRA, reason="pvcobra not installed")
@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
async def test_pvcobra_adetect_all_silence(mock_pvcobra):
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD
    from agent_platform.integrations.credentials import PicoVoiceCredentials
    from pydantic import SecretStr

    mock_handle = MagicMock()
    mock_handle.frame_length = 512
    mock_handle.process.return_value = 0.1
    mock_pvcobra.create.return_value = mock_handle

    vad = PvcobraVAD(PicoVoiceCredentials(access_key=SecretStr("mock_access_key")))

    async def _gen():
        yield _int16_chunk(512, 0)

    results = [span async for span in vad.adetect(_gen())]
    assert results == []


@pytest.mark.skipif(not _HAS_PVCOBRA, reason="pvcobra not installed")
@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
async def test_pvcobra_adetect_before_detect_does_not_raise(mock_pvcobra):
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD
    from agent_platform.integrations.credentials import PicoVoiceCredentials
    from pydantic import SecretStr

    mock_handle = MagicMock()
    mock_handle.frame_length = 512
    mock_handle.process.return_value = 0.1
    mock_pvcobra.create.return_value = mock_handle

    vad = PvcobraVAD(PicoVoiceCredentials(access_key=SecretStr("mock_access_key")))

    async def _gen():
        yield _int16_chunk(512, 0)

    results = [span async for span in vad.adetect(_gen())]
    assert results == []


@pytest.mark.skipif(not _HAS_TEN_VAD, reason="ten-vad not installed")
@patch("agent_platform.integrations.vad.ten.ten.TenVad")
async def test_ten_adetect_all_silence(mock_ten_vad_cls):
    mock_handle = MagicMock()
    mock_handle.process.return_value = (0.1, 0)
    mock_ten_vad_cls.return_value = mock_handle

    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()

    async def _gen():
        yield _int16_chunk(300, 0)

    results = [span async for span in vad.adetect(_gen())]
    assert results == []


@pytest.mark.skipif(not _HAS_TEN_VAD, reason="ten-vad not installed")
@patch("agent_platform.integrations.vad.ten.ten.TenVad")
async def test_ten_adetect_before_detect_does_not_raise_attribute_error(
    mock_ten_vad_cls,
):
    mock_handle = MagicMock()
    mock_handle.process.return_value = (0.1, 0)
    mock_ten_vad_cls.return_value = mock_handle

    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()
    assert vad.handle is None

    async def _gen():
        yield _int16_chunk(300, 0)

    results = [span async for span in vad.adetect(_gen(), config=TenVadConfig())]
    assert results == []


@patch("agent_platform.integrations.vad.silero.silero.load_silero_vad")
@patch("agent_platform.integrations.vad.silero.silero.get_speech_timestamps")
async def test_silero_adetect_buffer_flush(mock_get_speech_timestamps, mock_load):
    mock_load.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [{"start": 0, "end": 160}]

    from agent_platform.integrations.vad.silero.silero import SileroVAD

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

    results = [
        span async for span in vad.adetect(_gen(), config=SileroVadConfig())
    ]
    assert results == [SampleSpan(start=0, end=160)]
