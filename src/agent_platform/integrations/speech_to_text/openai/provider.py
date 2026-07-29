from __future__ import annotations

import io
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import ProviderError, error_logged, require_secret
from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.retry import with_retry
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.speech_to_text._base import buffered_stream
from agent_platform.integrations.speech_to_text.openai.config import OpenAIWhisperConfig
from agent_platform.integrations.speech_to_text.utils import parse_language


class OpenAIWhisperSTT(BaseSpeechToText[OpenAIWhisperConfig]):
    def __init__(self, credentials: OpenAICredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, OpenAICredentials)

    def _default_config(self) -> OpenAIWhisperConfig:
        return OpenAIWhisperConfig()

    def _build_client(self) -> AsyncOpenAI:
        api_key = require_secret(
            self._credentials.api_key, "OpenAI API key is required"
        )
        return AsyncOpenAI(api_key=api_key.get_secret_value())

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

        response_format = config.response_format
        if response_format is None:
            response_format = "verbose_json" if "whisper" in config.model else "json"

        create_kwargs: dict = dict(
            model=config.model,
            file=audio_file,
            language=config.language,
            temperature=config.temperature,
            response_format=response_format,
        )
        if config.timestamp_granularities and response_format == "verbose_json":
            create_kwargs["timestamp_granularities"] = config.timestamp_granularities
        if config.prompt:
            create_kwargs["prompt"] = config.prompt

        try:
            response = await client.audio.transcriptions.create(**create_kwargs)
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

        lang = parse_language(getattr(response, "language", "") or "")

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

        async def _transcribe(buffer: list[AudioChunk]) -> Transcript:
            combined = AudioChunk(
                data=b"".join(c.data for c in buffer),
                sample_rate=buffer[0].sample_rate,
                channels=buffer[0].channels,
                format=buffer[0].format,
                start=buffer[0].start,
                end=buffer[-1].end,
            )
            return await self.transcribe(combined, config)

        return buffered_stream(frames, config.min_duration_ms, _transcribe)
