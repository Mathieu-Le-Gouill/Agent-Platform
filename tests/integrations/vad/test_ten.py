from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("ten_vad")

from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import DataType
from agent_platform.core.schemas.span import SampleSpan
from agent_platform.integrations.vad.ten.config import TenVadConfig


def test_config_defaults():
    cfg = TenVadConfig()
    assert cfg.sample_rate == 16000
    assert cfg.threshold == 0.5


def test_provider_requirements(mocker):
    mocker.patch("agent_platform.integrations.vad.ten.ten.TenVad")
    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()
    req = vad.requirements
    assert req.sample_rates == (16000,)
    assert req.channels == 1
    assert req.dtype == DataType.INT16
    assert req.normalized is False


def test_detect_with_speech(mocker):
    mock_ten_vad_cls = mocker.patch("agent_platform.integrations.vad.ten.ten.TenVad")
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


def test_is_speech_above_threshold(mocker):
    mock_ten_vad_cls = mocker.patch("agent_platform.integrations.vad.ten.ten.TenVad")
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


def test_is_speech_below_threshold(mocker):
    mock_ten_vad_cls = mocker.patch("agent_platform.integrations.vad.ten.ten.TenVad")
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


async def test_adetect_async_flow(mocker):
    mock_ten_vad_cls = mocker.patch("agent_platform.integrations.vad.ten.ten.TenVad")
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


def test_detect_empty_returns_empty(mocker):
    mock_ten_vad_cls = mocker.patch("agent_platform.integrations.vad.ten.ten.TenVad")
    mock_ten_vad_cls.return_value = MagicMock()
    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()
    assert vad.detect([]) == []


def test_detect_all_silence_returns_empty(mocker):
    mock_ten_vad_cls = mocker.patch("agent_platform.integrations.vad.ten.ten.TenVad")
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


async def test_adetect_before_detect_does_not_raise_attribute_error(mocker):
    mock_ten_vad_cls = mocker.patch("agent_platform.integrations.vad.ten.ten.TenVad")
    mock_handle = MagicMock()
    mock_handle.process.return_value = (0.1, 0)
    mock_ten_vad_cls.return_value = mock_handle

    from agent_platform.integrations.vad.ten.ten import TenVAD

    # Regression test: adetect() previously never initialized self.handle,
    # so calling it before detect() raised AttributeError.
    vad = TenVAD()

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

    results = [span async for span in vad.adetect(_gen())]
    assert results == []
    mock_ten_vad_cls.assert_called_once_with(256, 0.5)


def test_handle_is_none_before_first_use(mocker):
    mock_ten_vad_cls = mocker.patch("agent_platform.integrations.vad.ten.ten.TenVad")
    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()
    assert vad.handle is None
    mock_ten_vad_cls.assert_not_called()


def test_ensure_handle_reuses_existing_handle_for_same_config(mocker):
    mock_ten_vad_cls = mocker.patch("agent_platform.integrations.vad.ten.ten.TenVad")
    mock_ten_vad_cls.return_value = MagicMock()

    from agent_platform.integrations.vad.ten.ten import TenVAD

    vad = TenVAD()
    config = TenVadConfig()

    first = vad._ensure_handle(config)
    second = vad._ensure_handle(config)

    assert first is second
    mock_ten_vad_cls.assert_called_once_with(config.hop_size, config.threshold)
