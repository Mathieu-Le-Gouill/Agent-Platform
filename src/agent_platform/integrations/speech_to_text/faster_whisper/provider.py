from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript
from agent_platform.integrations.speech_to_text._base import buffered_stream
from agent_platform.integrations.speech_to_text.faster_whisper.config import (
    FasterWhisperConfig,
)
from agent_platform.integrations.speech_to_text.faster_whisper.mappers import (
    map_utterances,
)
from agent_platform.integrations.speech_to_text.language import parse_language


class FasterWhisperSTT(BaseSpeechToText[FasterWhisperConfig]):
    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    def _default_config(self) -> FasterWhisperConfig:
        return FasterWhisperConfig()

    def _get_model(self, config: FasterWhisperConfig) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                config.model_size,
                device=config.device,
                compute_type=config.compute_type,
            )
        return self._model

    async def transcribe(
        self, audio: AudioChunk, config: FasterWhisperConfig | None = None
    ) -> Transcript:
        config = config or self._default_config()
        model = await asyncio.to_thread(self._get_model, config)
        audio_np = np.frombuffer(audio.data, dtype=np.float32)

        segments, info = await asyncio.to_thread(
            _run_transcribe, model, audio_np, config
        )

        utterances = map_utterances(segments, config)

        return Transcript(
            utterances=utterances,
            language=parse_language(info.language),
            metadata={"stt_provider": "faster-whisper", "model": config.model_size},
        )

    def stream(
        self,
        frames: AsyncIterator[AudioChunk],
        config: FasterWhisperConfig | None = None,
    ) -> AsyncIterator[Transcript]:
        config = config or self._default_config()

        async def _transcribe(chunks: list[AudioChunk]) -> Transcript:
            model = await asyncio.to_thread(self._get_model, config)
            buffer = [np.frombuffer(c.data, dtype=np.float32) for c in chunks]
            return await _transcribe_buffer(model, buffer, config)

        return buffered_stream(frames, config.min_duration_ms, _transcribe)


def _run_transcribe(
    model: WhisperModel, audio_np: np.ndarray, config: FasterWhisperConfig
) -> tuple[list[Segment], TranscriptionInfo]:
    segments, info = model.transcribe(
        audio_np,
        beam_size=config.beam_size,
        language=config.language,
        vad_filter=config.vad_filter,
        word_timestamps=config.word_timestamps,
        condition_on_previous_text=config.condition_on_previous_text,
    )
    return list(segments), info


async def _transcribe_buffer(
    model: WhisperModel,
    buffer: list[np.ndarray],
    config: FasterWhisperConfig,
) -> Transcript:
    audio_np = np.concatenate(buffer)
    segments, info = await asyncio.to_thread(_run_transcribe, model, audio_np, config)
    utterances = map_utterances(segments, config)
    return Transcript(
        utterances=utterances,
        language=parse_language(info.language),
        metadata={"stt_provider": "faster-whisper", "model": config.model_size},
    )
