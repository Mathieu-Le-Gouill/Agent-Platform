import pytest
from uuid import uuid4

from agent_platform.core.interfaces.vad.framebased import FrameBasedVAD
from agent_platform.core.interfaces.vad.config import VADConfig
from agent_platform.core.interfaces.vad.state import VADState
from agent_platform.core.interfaces.vad.requirements import AudioRequirements
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.span import SampleSpan
from agent_platform.core.schemas.enums import DataType


class _TestableVAD(FrameBasedVAD[VADConfig]):
    @property
    def requirements(self) -> AudioRequirements:
        return AudioRequirements(
            sample_rates=(16000,),
            channels=1,
            dtype=DataType.FLOAT32,
            normalized=True,
        )

    def _validate_chunk(
        self,
        chunk: AudioChunk,
        config_sample_rate: int,
    ) -> None:
        pass

    def _is_speech(self, chunk, config):
        return True

    def detect(self, audio_sequence, config=None):
        return []

    async def adetect(self, audio_sequence, config=None):
        return
        yield


@pytest.fixture
def vad():
    return _TestableVAD()


@pytest.fixture
def config():
    return VADConfig(
        speech_pad_ms=30, min_silence_duration_ms=100, min_speech_duration_ms=250
    )


def _chunk(start: int, end: int) -> AudioChunk:
    return AudioChunk(
        id=uuid4(),
        data=b"\x00\x80",
        sample_rate=16000,
        start=start,
        end=end,
        channels=1,
        dtype=DataType.FLOAT32,
    )


# --- _on_speech ---


def test_on_speech_initial_transition(vad, config):
    state = VADState()
    chunk = _chunk(0, 160)
    vad._on_speech(state, chunk, config)
    assert state.in_speech is True
    assert state.segment_start == 0
    assert state.speech_ms == 160
    assert state.silence_ms == 0


def test_on_speech_with_pad_when_last_speech_exists(vad, config):
    state = VADState(last_speech_end=500)
    chunk = _chunk(600, 760)
    vad._on_speech(state, chunk, config)
    assert state.segment_start == 570  # 600 - 30 pad
    assert state.in_speech is True
    assert state.speech_ms == 160


def test_on_speech_pad_clamped_to_zero(vad, config):
    state = VADState(last_speech_end=10)
    chunk = _chunk(20, 180)
    vad._on_speech(state, chunk, config)
    assert state.segment_start == 0  # max(0, 20 - 30)


def test_on_speech_already_in_speech(vad, config):
    state = VADState(in_speech=True, segment_start=100, speech_ms=300)
    chunk = _chunk(160, 320)
    vad._on_speech(state, chunk, config)
    assert state.in_speech is True
    assert state.segment_start == 100  # unchanged
    assert state.speech_ms == 460  # 300 + 160
    assert state.silence_ms == 0


def test_on_speech_accumulates_speech_ms(vad, config):
    state = VADState(in_speech=True, speech_ms=200)
    chunk = _chunk(100, 250)
    vad._on_speech(state, chunk, config)
    assert state.speech_ms == 350


# --- _on_silence ---


def test_on_silence_not_in_speech_resets_silence(vad, config):
    state = VADState(silence_ms=300)
    chunk = _chunk(500, 600)
    result = vad._on_silence(state, chunk, config)
    assert result is None
    assert state.silence_ms == 0


def test_on_silence_accumulates_below_threshold(vad, config):
    state = VADState(in_speech=True, speech_ms=300, segment_start=100)
    chunk = _chunk(200, 250)
    result = vad._on_silence(state, chunk, config)
    assert result is None
    assert state.silence_ms == 50
    assert state.in_speech is True


def test_on_silence_returns_span_when_speech_sufficient(vad, config):
    state = VADState(in_speech=True, speech_ms=300, segment_start=100)
    chunk = _chunk(400, 600)
    result = vad._on_silence(state, chunk, config)
    assert isinstance(result, SampleSpan)
    assert result.start == 100
    assert result.end == 430


def test_on_silence_resets_state_after_returning_span(vad, config):
    state = VADState(in_speech=True, speech_ms=300, segment_start=100)
    chunk = _chunk(400, 500)
    vad._on_silence(state, chunk, config)
    assert state.in_speech is False
    assert state.segment_start is None
    assert state.speech_ms == 0
    assert state.silence_ms == 0
    assert state.last_speech_end == 500


def test_on_silence_discards_when_speech_too_short(vad, config):
    state = VADState(in_speech=True, speech_ms=100, segment_start=50)
    chunk = _chunk(200, 350)
    result = vad._on_silence(state, chunk, config)
    assert result is None
    assert state.in_speech is False
    assert state.last_speech_end == 350


def test_on_silence_raises_when_segment_start_is_none(vad, config):
    state = VADState(in_speech=True, speech_ms=300, segment_start=None)
    chunk = _chunk(200, 350)
    with pytest.raises(RuntimeError, match="Invalid VAD state"):
        vad._on_silence(state, chunk, config)


def test_on_silence_resets_silence_on_subsequent_not_in_speech(vad, config):
    state = VADState(in_speech=True, speech_ms=300, segment_start=0)
    chunk_silence = _chunk(200, 350)
    vad._on_silence(state, chunk_silence, config)
    assert state.in_speech is False
    chunk2 = _chunk(400, 500)
    result = vad._on_silence(state, chunk2, config)
    assert result is None
    assert state.silence_ms == 0
