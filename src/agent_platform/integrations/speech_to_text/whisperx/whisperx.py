from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np
import whisperx

from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.integrations.speech_to_text.whisperx.config import WhisperXConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.core.schemas.enums import Language


class WhisperXSTT(BaseSpeechToText[WhisperXConfig]):
    def _default_config(self) -> WhisperXConfig:
        return WhisperXConfig()

    async def _load_model(self, config: WhisperXConfig) -> None:
        model_size = config.model_size
        device = config.device
        compute_type = config.compute_type

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: whisperx.load_model(
                model_size,
                device=device,
                compute_type=compute_type,
            ),
        )

    async def transcribe(
        self, audio: AudioChunk, config: WhisperXConfig | None = None
    ) -> Transcript:
        config = config or self._default_config()
        model = await self._load_model(config)

        audio_np = np.frombuffer(audio.data, dtype=np.float32).reshape(1, -1)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(audio_np, batch_size=config.batch_size),
        )

        language = _parse_language(result.get("language", ""))

        utterances = [
            Utterance(
                text=seg["text"].strip(),
                start_ms=int(seg.get("start", 0) * 1000),
                end_ms=int(seg.get("end", 0) * 1000),
                confidence=seg.get("confidence"),
            )
            for seg in result.get("segments", [])
        ]

        return Transcript(
            utterances=utterances,
            language=language,
            metadata={"stt_provider": "whisperx", "model": config.model_size},
        )

    def stream(
        self, frames: AsyncIterator[AudioChunk], config: WhisperXConfig | None = None
    ) -> AsyncIterator[Transcript]:
        config = config or self._default_config()

        async def _stream() -> AsyncIterator[Transcript]:
            model = await self._load_model(config)

            buffer: list[AudioChunk] = []
            buffer_ms = 0

            async for chunk in frames:
                buffer.append(chunk)
                buffer_ms += chunk.end - chunk.start

                if buffer_ms < config.min_duration_ms:
                    continue

                yield await self._transcribe_buffer(model, buffer, config)
                buffer.clear()
                buffer_ms = 0

            if buffer:
                yield await self._transcribe_buffer(model, buffer, config)

        return _stream()

    async def _transcribe_buffer(
        self, model, buffer: list[AudioChunk], config: WhisperXConfig
    ) -> Transcript:
        audio_np = np.concatenate(
            [np.frombuffer(c.data, dtype=np.float32) for c in buffer]
        ).reshape(1, -1)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(audio_np, batch_size=config.batch_size),
        )

        language = _parse_language(result.get("language", ""))
        utterances = [
            Utterance(
                text=seg["text"].strip(),
                start_ms=int(seg.get("start", 0) * 1000),
                end_ms=int(seg.get("end", 0) * 1000),
                confidence=seg.get("confidence"),
            )
            for seg in result.get("segments", [])
        ]

        return Transcript(
            utterances=utterances,
            language=language,
            metadata={"stt_provider": "whisperx", "model": config.model_size},
        )


def _parse_language(code: str) -> Language | None:
    try:
        return Language(code)
    except ValueError:
        return None
