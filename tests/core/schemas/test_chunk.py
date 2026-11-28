import pytest

from agent_platform.core.schemas.chunk import AudioChunk


class TestAudioChunkDuration:
    @pytest.mark.parametrize(
        ("start", "end", "sample_rate", "expected"),
        [
            (0, 16000, 16000, 1000.0),
            (8000, 8480, 16000, 30.0),
            (0, 441, 44100, 10.0),
            (0, 0, 16000, 0.0),
        ],
    )
    def test_duration_ms_converts_sample_offsets(
        self, start, end, sample_rate, expected
    ):
        chunk = AudioChunk(start=start, end=end, sample_rate=sample_rate)

        assert chunk.duration_ms == pytest.approx(expected)

    def test_duration_ms_is_not_serialized(self):
        assert "duration_ms" not in AudioChunk(end=160).model_dump()
