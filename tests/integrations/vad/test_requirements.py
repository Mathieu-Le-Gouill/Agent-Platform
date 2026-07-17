import pytest

from agent_platform.core.schemas.enums import DataType
from agent_platform.core.interfaces.vad.requirements import AudioRequirements


class TestAudioRequirements:
    def test_can_create_with_sample_rates_channels_dtype_normalized(self):
        req = AudioRequirements(
            sample_rates=(8000, 16000),
            channels=1,
            dtype=DataType.FLOAT32,
            normalized=True,
        )
        assert req.sample_rates == (8000, 16000)
        assert req.channels == 1
        assert req.dtype == DataType.FLOAT32
        assert req.normalized is True

    def test_validate_passes_when_chunk_matches_requirements(self):
        req = AudioRequirements(
            sample_rates=(8000, 16000),
            channels=1,
            dtype=DataType.FLOAT32,
            normalized=True,
        )
        chunk = _make_mock_chunk(sample_rate=16000, channels=1, dtype=DataType.FLOAT32)
        req.validate(chunk)

    def test_validate_raises_value_error_for_unsupported_sample_rate(self):
        req = AudioRequirements(
            sample_rates=(8000, 16000),
            channels=1,
            dtype=DataType.FLOAT32,
            normalized=True,
        )
        chunk = _make_mock_chunk(sample_rate=44100, channels=1, dtype=DataType.FLOAT32)
        with pytest.raises(ValueError, match="Unsupported sample rate"):
            req.validate(chunk)

    def test_validate_raises_value_error_for_mismatched_channels(self):
        req = AudioRequirements(
            sample_rates=(8000, 16000),
            channels=1,
            dtype=DataType.FLOAT32,
            normalized=True,
        )
        chunk = _make_mock_chunk(sample_rate=16000, channels=2, dtype=DataType.FLOAT32)
        with pytest.raises(ValueError, match="Expected 1 channel"):
            req.validate(chunk)

    def test_validate_raises_type_error_for_mismatched_dtype(self):
        req = AudioRequirements(
            sample_rates=(8000, 16000),
            channels=1,
            dtype=DataType.FLOAT32,
            normalized=True,
        )
        chunk = _make_mock_chunk(sample_rate=16000, channels=1, dtype=DataType.INT16)
        with pytest.raises(TypeError, match="Expected dtype FLOAT32"):
            req.validate(chunk)


class _MockChunk:
    def __init__(self, sample_rate, channels, dtype):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype


def _make_mock_chunk(sample_rate, channels, dtype):
    return _MockChunk(sample_rate=sample_rate, channels=channels, dtype=dtype)
