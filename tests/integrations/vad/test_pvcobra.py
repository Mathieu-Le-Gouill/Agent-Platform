import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from pydantic import SecretStr

pytest.importorskip("pvcobra")

from agent_platform.integrations.vad.pvcobra.config import PvcobraVadConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import DataType
from agent_platform.core.errors import ProviderError
from agent_platform.integrations.credentials import PicoVoiceCredentials

FRAME_LENGTH = 512


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


def _mock_handle(**process_kwargs) -> MagicMock:
    handle = MagicMock()
    handle.frame_length = FRAME_LENGTH
    if process_kwargs:
        handle.process.configure_mock(**process_kwargs)
    return handle


def _credentials() -> PicoVoiceCredentials:
    return PicoVoiceCredentials(access_key=SecretStr("mock_access_key"))


def test_config_defaults():
    cfg = PvcobraVadConfig()
    assert cfg.sample_rate == 16000
    assert cfg.threshold == 0.5


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_provider_requirements(mock_pvcobra):
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    mock_pvcobra.create.return_value = _mock_handle()
    vad = PvcobraVAD(_credentials())
    req = vad.requirements
    assert req.sample_rates == (16000,)
    assert req.channels == 1
    assert req.dtype == DataType.INT16


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_requirements_rejects_invalid_sample_rate(mock_pvcobra):
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    mock_pvcobra.create.return_value = _mock_handle()
    vad = PvcobraVAD(_credentials())
    chunk = _chunk(FRAME_LENGTH, 0, sample_rate=10000)
    with pytest.raises(ValueError, match="Unsupported sample rate"):
        vad.requirements.validate(chunk)


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_requirements_rejects_invalid_dtype(mock_pvcobra):
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    mock_pvcobra.create.return_value = _mock_handle()
    vad = PvcobraVAD(_credentials())
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


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_requirements_rejects_invalid_channels(mock_pvcobra):
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    mock_pvcobra.create.return_value = _mock_handle()
    vad = PvcobraVAD(_credentials())
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


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_requirements_accepts_valid(mock_pvcobra):
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    mock_pvcobra.create.return_value = _mock_handle()
    vad = PvcobraVAD(_credentials())
    chunk = _chunk(FRAME_LENGTH, 0)
    vad.requirements.validate(chunk)


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_detect_with_speech_spans(mock_pvcobra):
    mock_handle = _mock_handle(side_effect=[0.9, 0.8, 0.1])
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())

    chunks = [
        _chunk(FRAME_LENGTH, 0),
        _chunk(FRAME_LENGTH, FRAME_LENGTH),
        _chunk(FRAME_LENGTH, 2 * FRAME_LENGTH),
    ]

    result = vad.detect(chunks)
    assert len(result) == 1


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_is_speech_above_threshold(mock_pvcobra):
    mock_handle = _mock_handle(return_value=0.9)
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())
    vad._ensure_handle(PvcobraVadConfig())

    chunk = _chunk(FRAME_LENGTH, 0)
    config = PvcobraVadConfig()
    assert vad._is_speech(chunk, config) is True
    mock_handle.process.assert_called_once_with(chunk.data)


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_is_speech_below_threshold(mock_pvcobra):
    mock_handle = _mock_handle(return_value=0.1)
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())
    vad._ensure_handle(PvcobraVadConfig())

    chunk = _chunk(FRAME_LENGTH, 0)
    config = PvcobraVadConfig()
    assert vad._is_speech(chunk, config) is False


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_is_speech_wrong_frame_length_raises_provider_error(mock_pvcobra):
    mock_handle = _mock_handle()
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())
    vad._ensure_handle(PvcobraVadConfig())

    chunk = _chunk(160, 0)
    config = PvcobraVadConfig()
    with pytest.raises(ProviderError, match="512"):
        vad._is_speech(chunk, config)
    mock_handle.process.assert_not_called()


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_is_speech_native_error_translated_to_provider_error(mock_pvcobra):
    mock_handle = _mock_handle()
    mock_handle.process.side_effect = ValueError("native failure")
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())
    vad._ensure_handle(PvcobraVadConfig())

    chunk = _chunk(FRAME_LENGTH, 0)
    config = PvcobraVadConfig()
    with pytest.raises(ProviderError, match="native failure"):
        vad._is_speech(chunk, config)


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_is_speech_raises_runtime_error_when_handle_not_initialized(mock_pvcobra):
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())

    chunk = _chunk(FRAME_LENGTH, 0)
    with pytest.raises(RuntimeError, match="not initialized"):
        vad._is_speech(chunk, PvcobraVadConfig())


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_close_deletes_native_handle(mock_pvcobra):
    mock_handle = _mock_handle()
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())
    vad._ensure_handle(PvcobraVadConfig())

    vad.close()

    mock_handle.delete.assert_called_once()
    assert vad._handle is None


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_close_is_noop_when_handle_never_created(mock_pvcobra):
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())
    vad.close()
    assert vad._handle is None


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_del_calls_close_without_raising(mock_pvcobra):
    mock_handle = _mock_handle()
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())
    vad._ensure_handle(PvcobraVadConfig())

    vad.__del__()

    mock_handle.delete.assert_called_once()


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_detect_empty_returns_empty(mock_pvcobra):
    mock_pvcobra.create.return_value = _mock_handle()
    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())
    assert vad.detect([]) == []


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
def test_detect_no_span_when_silence_before_speech(mock_pvcobra):
    mock_handle = _mock_handle(side_effect=[0.1, 0.1])
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())

    chunks = [
        _chunk(FRAME_LENGTH, 0),
        _chunk(FRAME_LENGTH, FRAME_LENGTH),
    ]

    result = vad.detect(chunks)
    assert result == []


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
async def test_adetect_yields_span(mock_pvcobra):
    mock_handle = _mock_handle(side_effect=[0.9, 0.1])
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())

    async def _gen():
        yield _chunk(FRAME_LENGTH, 0)
        yield _chunk(FRAME_LENGTH, FRAME_LENGTH)

    results = [span async for span in vad.adetect(_gen())]
    assert len(results) == 1


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
async def test_adetect_before_detect_does_not_raise_attribute_error(mock_pvcobra):
    mock_handle = _mock_handle(return_value=0.1)
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())

    async def _gen():
        yield _chunk(FRAME_LENGTH, 0)

    results = [span async for span in vad.adetect(_gen())]
    assert results == []


@patch("agent_platform.integrations.vad.pvcobra.pvcobra.pvcobra")
async def test_adetect_validate_chunk_raises(mock_pvcobra):
    mock_handle = _mock_handle(return_value=0.1)
    mock_pvcobra.create.return_value = mock_handle

    from agent_platform.integrations.vad.pvcobra.pvcobra import PvcobraVAD

    vad = PvcobraVAD(_credentials())

    async def _gen():
        yield _chunk(FRAME_LENGTH, 0, sample_rate=8000)

    with pytest.raises((ValueError,), match="Unsupported sample rate"):
        async for _ in vad.adetect(_gen()):
            pass
