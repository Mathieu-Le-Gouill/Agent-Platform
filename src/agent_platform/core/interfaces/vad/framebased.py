from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Generic

from agent_platform.core.interfaces.vad.base import BaseVAD, ConfigT
from agent_platform.core.interfaces.vad.state import VADState
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.span import SampleSpan


def _ms_to_samples(ms: int, sample_rate: int) -> int:
    return ms * sample_rate // 1000


class FrameBasedVAD(BaseVAD[ConfigT], Generic[ConfigT]):
    """Runs `_is_speech` on each frame and turns the speech/silence sequence
    into padded spans. Subclasses supply `_is_speech`, `_default_config` and,
    when the backend needs setup per config, `_prepare`."""

    @abstractmethod
    def _default_config(self) -> ConfigT: ...

    def _prepare(self, config: ConfigT) -> None:
        """Set the backend up for `config` before the first frame."""

    def detect(
        self,
        audio_sequence: Sequence[AudioChunk],
        config: ConfigT | None = None,
    ) -> list[SampleSpan]:
        config = config or self._default_config()
        self._prepare(config)
        state = VADState()
        spans: list[SampleSpan] = []
        for chunk in audio_sequence:
            span = self._step(state, chunk, config)
            if span is not None:
                spans.append(span)
        return spans

    async def adetect(
        self,
        audio_sequence: AsyncIterator[AudioChunk],
        config: ConfigT | None = None,
    ) -> AsyncIterator[SampleSpan]:
        config = config or self._default_config()
        self._prepare(config)
        state = VADState()
        async for chunk in audio_sequence:
            span = self._step(state, chunk, config)
            if span is not None:
                yield span

    def _step(
        self, state: VADState, chunk: AudioChunk, config: ConfigT
    ) -> SampleSpan | None:
        self._validate_chunk(chunk, config.sample_rate)
        if self._is_speech(chunk, config):
            self._on_speech(state, chunk, config)
            return None
        return self._on_silence(state, chunk, config)

    def _on_speech(
        self,
        state: VADState,
        chunk: AudioChunk,
        config: ConfigT,
    ) -> None:

        state.silence_ms = 0
        state.speech_ms += chunk.duration_ms

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

        state.silence_ms += chunk.duration_ms

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
