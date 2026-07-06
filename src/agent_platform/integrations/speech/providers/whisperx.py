from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np
import whisperx

from agent_platform.integrations.speech.base import BaseSpeechToText
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.conversation import Transcript, Utterance
from agent_platform.models.enums import Language


class WhisperXSTT(BaseSpeechToText):

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "float32",
        batch_size: int = 16,
        min_duration_ms: int = 5000,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._batch_size = batch_size
        self._min_duration_ms = min_duration_ms
        self._model = None

    async def _load_model(self) -> None:
        if self._model is not None:
            return

        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: whisperx.load_model(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            ),
        )

    async def transcribe(self, audio: AudioChunk) -> Transcript:

        await self._load_model()
        model = self._model

        if model is None:
            raise RuntimeError("Failed to load WhisperX Speech To Text model")

        audio_np = np.frombuffer(audio.data, dtype=np.float32).reshape(1, -1)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(audio_np, batch_size=self._batch_size),
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
            metadata={"stt_provider": "whisperx", "model": self._model_size},
        )

    async def stream(
        self, frames: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcript]:
        
        await self._load_model()

        buffer: list[AudioChunk] = []
        buffer_ms = 0

        async for chunk in frames:
            buffer.append(chunk)
            buffer_ms += chunk.end - chunk.start

            if buffer_ms < self._min_duration_ms:
                continue

            yield await self._transcribe_buffer(buffer)
            buffer.clear()
            buffer_ms = 0

        if buffer:
            yield await self._transcribe_buffer(buffer)

    async def _transcribe_buffer(
        self, buffer: list[AudioChunk]
    ) -> Transcript:
        
        model = self._model

        if model is None:
            raise RuntimeError("Failed to load WhisperX Speech To Text model")

        audio_np = np.concatenate(
            [np.frombuffer(c.data, dtype=np.float32) for c in buffer]
        ).reshape(1, -1)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(audio_np, batch_size=self._batch_size),
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
            metadata={"stt_provider": "whisperx", "model": self._model_size},
        )


def _parse_language(code: str) -> Language | None:
    try:
        return Language(code)
    except ValueError:
        return None
