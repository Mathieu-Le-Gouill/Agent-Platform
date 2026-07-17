from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np
from faster_whisper import WhisperModel

from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.integrations.speech_to_text.faster_whisper.config import (
    FasterWhisperConfig,
)
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.core.schemas.enums import Language


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
            model.transcribe,
            audio_np,
            beam_size=config.beam_size,
            language=config.language,
            vad_filter=config.vad_filter,
        )

        utterances = [
            Utterance(
                text=seg.text.strip(),
                start_ms=int(seg.start * 1000),
                end_ms=int(seg.end * 1000),
            )
            for seg in segments
        ]

        return Transcript(
            utterances=utterances,
            language=_parse_language(info.language),
            metadata={"stt_provider": "faster-whisper", "model": config.model_size},
        )

    def stream(
        self,
        frames: AsyncIterator[AudioChunk],
        config: FasterWhisperConfig | None = None,
    ) -> AsyncIterator[Transcript]:
        config = config or self._default_config()

        async def _stream() -> AsyncIterator[Transcript]:
            model = await asyncio.to_thread(self._get_model, config)
            buffer: list[np.ndarray] = []
            buffer_ms = 0

            async for chunk in frames:
                buffer.append(np.frombuffer(chunk.data, dtype=np.float32))
                buffer_ms += chunk.end - chunk.start

                if buffer_ms < config.min_duration_ms:
                    continue

                yield await _transcribe_buffer(model, buffer, config)
                buffer.clear()
                buffer_ms = 0

            if buffer:
                yield await _transcribe_buffer(model, buffer, config)

        return _stream()


async def _transcribe_buffer(
    model: WhisperModel,
    buffer: list[np.ndarray],
    config: FasterWhisperConfig,
) -> Transcript:
    audio_np = np.concatenate(buffer)
    segments, info = await asyncio.to_thread(
        model.transcribe,
        audio_np,
        beam_size=config.beam_size,
        language=config.language,
        vad_filter=config.vad_filter,
    )
    utterances = [
        Utterance(
            text=seg.text.strip(),
            start_ms=int(seg.start * 1000),
            end_ms=int(seg.end * 1000),
        )
        for seg in segments
    ]
    return Transcript(
        utterances=utterances,
        language=_parse_language(info.language),
        metadata={"stt_provider": "faster-whisper", "model": config.model_size},
    )


def _parse_language(code: str) -> Language | None:
    try:
        return Language(code)
    except ValueError:
        return None
