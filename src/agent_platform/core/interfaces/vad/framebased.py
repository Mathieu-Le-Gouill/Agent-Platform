from __future__ import annotations

from typing import Generic

from agent_platform.core.interfaces.vad.base import BaseVAD, ConfigT
from agent_platform.core.schemas.span import SampleSpan
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.interfaces.vad.state import VADState


class FrameBasedVAD(BaseVAD[ConfigT], Generic[ConfigT]):
    def _on_speech(
        self,
        state: VADState,
        chunk: AudioChunk,
        config: ConfigT,
    ) -> None:

        state.silence_ms = 0
        state.speech_ms += chunk.end - chunk.start

        if state.in_speech:
            return

        state.in_speech = True
        state.segment_start = chunk.start

        if state.last_speech_end is not None:
            state.segment_start = max(
                0,
                state.segment_start - config.speech_pad_ms,
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

        state.silence_ms += chunk.end - chunk.start

        if state.silence_ms < config.min_silence_duration_ms:
            return None

        span = None

        if state.speech_ms >= config.min_speech_duration_ms:
            if state.segment_start is None:
                raise RuntimeError("Invalid VAD state")

            span = SampleSpan(
                start=state.segment_start,
                end=chunk.start + config.speech_pad_ms,
            )

        state.in_speech = False
        state.segment_start = None
        state.speech_ms = 0
        state.silence_ms = 0
        state.last_speech_end = chunk.end

        return span
