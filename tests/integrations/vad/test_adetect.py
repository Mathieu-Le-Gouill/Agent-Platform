import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from agent_platform.integrations.vad.configuration import (
    WebrtcVadConfig,
    PvcobraVadConfig,
    SileroVadConfig,
    TenVadConfig,
)
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.span import SampleSpan
from agent_platform.models.enums import DataType


@patch("agent_platform.integrations.vad.providers.webrtc.webrtcvad")
async def test_webrtc_adetect_yields_span(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_vad.is_speech.side_effect = [True, True, False]
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
            data=b"\x00" * 300,
            sample_rate=16000,
            start=300,
            end=600,
            channels=1,
            dtype=DataType.INT16,
        )
        yield AudioChunk(
            id=uuid4(),
            data=b"\x00" * 150,
            sample_rate=16000,
            start=600,
            end=750,
            channels=1,
            dtype=DataType.INT16,
        )

    results = [span async for span in vad.adetect(_gen())]
    assert len(results) == 1
    assert results[0] == SampleSpan(start=0, end=630)


@patch("agent_platform.integrations.vad.providers.webrtc.webrtcvad")
async def test_webrtc_adetect_all_silence(mock_webrtcvad):
    mock_vad = MagicMock()
    mock_vad.is_speech.return_value = False
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

    results = [span async for span in vad.adetect(_gen())]
    assert results == []


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
async def test_pvcobra_adetect_yields_span(mock_pvcobra):
    mock_handle = MagicMock()
    mock_handle.process.side_effect = [0.9, 0.1]
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD
    from pydantic import SecretStr

    vad = PvcobraVAD(SecretStr("mock_access_key"))

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


@patch("agent_platform.integrations.vad.providers.pvcobra.pvcobra")
async def test_pvcobra_adetect_all_silence(mock_pvcobra):
    mock_handle = MagicMock()
    mock_handle.process.return_value = 0.1
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.providers.pvcobra import PvcobraVAD
    from pydantic import SecretStr

    vad = PvcobraVAD(SecretStr("mock_access_key"))

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

    results = [span async for span in vad.adetect(_gen())]
    assert results == []


@patch("agent_platform.integrations.vad.providers.ten.TenVad")
async def test_ten_adetect_all_silence(mock_ten_vad_cls):
    mock_handle = MagicMock()
    mock_handle.process.return_value = (0.1, 0)
    mock_ten_vad_cls.return_value = mock_handle

    from agent_platform.integrations.vad.providers.ten import TenVAD

    vad = TenVAD()
    vad.handle = mock_handle

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

    results = [span async for span in vad.adetect(_gen())]
    assert results == []


@patch("agent_platform.integrations.vad.providers.silero.load_silero_vad")
@patch("agent_platform.integrations.vad.providers.silero.get_speech_timestamps")
async def test_silero_adetect_buffer_flush(mock_get_speech_timestamps, mock_load):
    mock_load.return_value = MagicMock()
    mock_get_speech_timestamps.return_value = [{"start": 0, "end": 160}]

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

    results = [span async for span in vad.adetect(_gen())]
    assert results == [SampleSpan(start=0, end=160)]
