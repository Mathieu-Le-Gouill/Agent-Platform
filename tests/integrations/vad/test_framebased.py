from uuid import uuid4

import pytest

from agent_platform.core.interfaces.vad.config import VADConfig
from agent_platform.core.interfaces.vad.framebased import FrameBasedVAD
from agent_platform.core.interfaces.vad.requirements import AudioRequirements
from agent_platform.core.interfaces.vad.state import VADState
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import DataType
from agent_platform.core.schemas.span import SampleSpan

_SPEECH = b"\x01\x00"
_SILENCE = b"\x00\x80"


class _TestableVAD(FrameBasedVAD[VADConfig]):
    def __init__(self) -> None:
        self.prepared: list[VADConfig] = []

    def _default_config(self) -> VADConfig:
        return VADConfig(
            speech_pad_ms=30, min_silence_duration_ms=100, min_speech_duration_ms=250
        )

    def _prepare(self, config: VADConfig) -> None:
        self.prepared.append(config)

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
        return chunk.data == _SPEECH


@pytest.fixture
def vad():
    return _TestableVAD()


@pytest.fixture
def config():
    return VADConfig(
        speech_pad_ms=30, min_silence_duration_ms=100, min_speech_duration_ms=250
    )


def _chunk(
    start: int, end: int, sample_rate: int = 1000, data: bytes = _SILENCE
) -> AudioChunk:
    return AudioChunk(
        id=uuid4(),
        data=data,
        sample_rate=sample_rate,
        start=start,
        end=end,
        channels=1,
        dtype=DataType.FLOAT32,
    )


def _speech_then_silence() -> list[AudioChunk]:
    # 300 ms of speech, then 200 ms of silence, at 1 kHz
    return [_chunk(0, 300, data=_SPEECH), _chunk(300, 500)]


async def _aiter(chunks):
    for chunk in chunks:
        yield chunk


# --- detect / adetect ---


def test_detect_turns_speech_then_silence_into_a_padded_span(vad):
    assert vad.detect(_speech_then_silence()) == [SampleSpan(start=0, end=330)]


async def test_adetect_yields_the_same_spans_as_detect(vad):
    spans = [span async for span in vad.adetect(_aiter(_speech_then_silence()))]
    assert spans == vad.detect(_speech_then_silence())


@pytest.mark.parametrize("mode", ["detect", "adetect"])
async def test_the_backend_is_prepared_once_with_the_given_config(vad, config, mode):
    if mode == "detect":
        vad.detect(_speech_then_silence(), config)
    else:
        [span async for span in vad.adetect(_aiter(_speech_then_silence()), config)]
    assert vad.prepared == [config]


def test_detect_on_no_audio_returns_no_span_and_uses_the_default_config(vad):
    assert vad.detect([]) == []
    assert vad.prepared == [vad._default_config()]


def test_speech_still_open_at_the_end_yields_no_span(vad):
    assert vad.detect([_chunk(0, 300, data=_SPEECH)]) == []


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


# --- sample offsets vs milliseconds ---


def test_on_speech_counts_milliseconds_not_samples_at_16khz(vad, config):
    state = VADState()
    vad._on_speech(state, _chunk(0, 512, sample_rate=16000), config)
    assert state.speech_ms == 32


def test_on_speech_accumulates_fractional_ms_without_drift(vad, config):
    state = VADState()
    for i in range(100):
        vad._on_speech(state, _chunk(i * 300, (i + 1) * 300, sample_rate=16000), config)
    assert state.speech_ms == pytest.approx(1875)  # 100 x 18.75 ms


def test_on_speech_pads_backward_in_samples_at_16khz(vad, config):
    state = VADState(last_speech_end=8000)
    vad._on_speech(state, _chunk(16000, 16512, sample_rate=16000), config)
    assert state.segment_start == 16000 - 480  # 30 ms at 16 kHz


def test_on_silence_counts_milliseconds_and_pads_end_in_samples_at_8khz(vad, config):
    state = VADState(in_speech=True, speech_ms=300, segment_start=0)
    first = vad._on_silence(state, _chunk(8000, 8400, sample_rate=8000), config)
    assert first is None
    assert state.silence_ms == 50
    span = vad._on_silence(state, _chunk(8400, 8800, sample_rate=8000), config)
    assert span == SampleSpan(start=0, end=8400 + 240)  # 30 ms at 8 kHz
