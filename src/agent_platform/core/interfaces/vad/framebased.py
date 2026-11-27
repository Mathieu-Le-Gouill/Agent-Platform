from __future__ import annotations

from typing import Generic

from agent_platform.core.interfaces.vad.base import BaseVAD, ConfigT
from agent_platform.core.interfaces.vad.state import VADState
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.span import SampleSpan


def _duration_ms(chunk: AudioChunk) -> float:
    return (chunk.end - chunk.start) * 1000 / chunk.sample_rate


def _ms_to_samples(ms: int, sample_rate: int) -> int:
    return ms * sample_rate // 1000


class FrameBasedVAD(BaseVAD[ConfigT], Generic[ConfigT]):
    def _on_speech(
        self,
        state: VADState,
        chunk: AudioChunk,
        config: ConfigT,
    ) -> None:

        state.silence_ms = 0
        state.speech_ms += _duration_ms(chunk)

        if state.in_speech:
            return

        state.in_speech = True
        state.segment_start = chunk.start

        if state.last_speech_end is not None:
            # only pad backward for segments after the first (nothing precedes the very first one)
            state.segment_start = max(
                0,
                state.segment_start
                - _ms_to_samples(config.speech_pad_ms, chunk.sample_rate),
            )

    def _on_silence(
        self,
        state: VADState,
        chunk: AudioChunk,
        config: ConfigT,
    ) -> SampleSpan | None:

        if not state.in_speech:
            state.silence_ms = 0
            return None

        state.silence_ms += _duration_ms(chunk)

        if state.silence_ms < config.min_silence_duration_ms:
            return None

        span = None

        if state.speech_ms >= config.min_speech_duration_ms:
            if state.segment_start is None:
                raise RuntimeError("Invalid VAD state")

            span = SampleSpan(
                start=state.segment_start,
                end=chunk.start
                + _ms_to_samples(config.speech_pad_ms, chunk.sample_rate),
            )

        state.in_speech = False
        state.segment_start = None
        state.speech_ms = 0
        state.silence_ms = 0
        state.last_speech_end = chunk.end

        return span
