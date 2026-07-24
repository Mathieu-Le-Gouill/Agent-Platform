from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.integrations.speech_to_text.faster_whisper.config import (
    FasterWhisperConfig,
)
from agent_platform.integrations.speech_to_text.utils import parse_language


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

        utterances = _map_utterances(segments, config)

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


def _map_utterances(
    segments: list[Segment], config: FasterWhisperConfig
) -> list[Utterance]:
    if config.word_timestamps:
        utterances = []
        for seg in segments:
            if not seg.words:
                utterances.append(
                    Utterance(
                        text=seg.text.strip(),
                        start_ms=int(seg.start * 1000),
                        end_ms=int(seg.end * 1000),
                    )
                )
                continue
            for word in seg.words:
                utterances.append(
                    Utterance(
                        text=word.word.strip(),
                        start_ms=int(word.start * 1000),
                        end_ms=int(word.end * 1000),
                        confidence=word.probability,
                    )
                )
        return utterances

    return [
        Utterance(
            text=seg.text.strip(),
            start_ms=int(seg.start * 1000),
            end_ms=int(seg.end * 1000),
        )
        for seg in segments
    ]


async def _transcribe_buffer(
    model: WhisperModel,
    buffer: list[np.ndarray],
    config: FasterWhisperConfig,
) -> Transcript:
    audio_np = np.concatenate(buffer)
    segments, info = await asyncio.to_thread(_run_transcribe, model, audio_np, config)
    utterances = _map_utterances(segments, config)
    return Transcript(
        utterances=utterances,
        language=parse_language(info.language),
        metadata={"stt_provider": "faster-whisper", "model": config.model_size},
    )
