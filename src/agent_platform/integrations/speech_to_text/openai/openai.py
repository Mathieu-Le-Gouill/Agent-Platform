from __future__ import annotations

import asyncio
import io
from typing import AsyncIterator

import numpy as np
from openai import AsyncOpenAI

from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.integrations.speech_to_text.openai.config import OpenAIWhisperConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.core.schemas.enums import Language
from agent_platform.core.errors import (
    MissingCredentialError,
    ProviderError,
    error_logged,
    with_retry,
)


class OpenAIWhisperSTT(BaseSpeechToText[OpenAIWhisperConfig]):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else OpenAICredentials()
        )

    def _default_config(self) -> OpenAIWhisperConfig:
        return OpenAIWhisperConfig()

    def _build_client(self) -> AsyncOpenAI:
        if not self._credentials.api_key:
            raise MissingCredentialError("OpenAI API key is required")
        return AsyncOpenAI(api_key=self._credentials.api_key.get_secret_value())

    @error_logged(re_raise=ProviderError, message="Speech-to-text failed")
    @with_retry()
    async def transcribe(
        self, audio: AudioChunk, config: OpenAIWhisperConfig | None = None
    ) -> Transcript:
        config = config or self._default_config()
        client = self._build_client()
        fmt = audio.format.value if audio.format else "wav"
        audio_file = io.BytesIO(audio.data)
        audio_file.name = f"audio.{fmt}"

        try:
            response = await client.audio.transcriptions.create(
                model=config.model,
                file=audio_file,
                language=config.language,
                temperature=config.temperature,
                response_format="verbose_json",
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI Whisper transcription failed: {exc}") from exc

        utterances: list[Utterance] = []
        if hasattr(response, "segments") and response.segments:
            utterances = [
                Utterance(
                    text=seg.text.strip(),
                    start_ms=int(seg.start * 1000),
                    end_ms=int(seg.end * 1000),
                )
                for seg in response.segments
            ]
        else:
            utterances = [Utterance(text=response.text)]

        lang = _parse_language(getattr(response, "language", "") or "")

        return Transcript(
            utterances=utterances,
            language=lang,
            metadata={"stt_provider": "openai-whisper", "model": config.model},
        )

    def stream(
        self,
        frames: AsyncIterator[AudioChunk],
        config: OpenAIWhisperConfig | None = None,
    ) -> AsyncIterator[Transcript]:
        config = config or self._default_config()

        async def _stream() -> AsyncIterator[Transcript]:
            buffer: list[AudioChunk] = []
            buffer_ms = 0

            async for chunk in frames:
                buffer.append(chunk)
                buffer_ms += chunk.end - chunk.start

                if buffer_ms < config.min_duration_ms:
                    continue

                combined = AudioChunk(
                    data=b"".join(c.data for c in buffer),
                    sample_rate=buffer[0].sample_rate,
                    channels=buffer[0].channels,
                    format=buffer[0].format,
                    start=buffer[0].start,
                    end=buffer[-1].end,
                )
                yield await self.transcribe(combined, config)
                buffer.clear()
                buffer_ms = 0

            if buffer:
                combined = AudioChunk(
                    data=b"".join(c.data for c in buffer),
                    sample_rate=buffer[0].sample_rate,
                    channels=buffer[0].channels,
                    format=buffer[0].format,
                    start=buffer[0].start,
                    end=buffer[-1].end,
                )
                yield await self.transcribe(combined, config)

        return _stream()


def _parse_language(code: str) -> Language | None:
    try:
        return Language(code)
    except ValueError:
        return None
