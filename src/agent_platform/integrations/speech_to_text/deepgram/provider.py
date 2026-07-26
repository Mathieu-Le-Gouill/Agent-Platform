from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from agent_platform.core.credentials import resolve_credentials
from agent_platform.core.errors import (
    ProviderError,
    error_logged,
    require_secret,
    with_retry,
)
from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.integrations.credentials import DeepgramCredentials
from agent_platform.integrations.speech_to_text.deepgram.config import DeepgramConfig
from agent_platform.integrations.speech_to_text.utils import parse_language


class DeepgramSTT(BaseSpeechToText[DeepgramConfig]):
    def __init__(self, credentials: DeepgramCredentials | None = None) -> None:
        self._credentials = resolve_credentials(credentials, DeepgramCredentials)

    def _default_config(self) -> DeepgramConfig:
        return DeepgramConfig()

    def _api_key(self) -> str:
        api_key = require_secret(
            self._credentials.api_key, "Deepgram API key is required"
        )
        return api_key.get_secret_value()

    def _build_client(self):
        from deepgram import AsyncDeepgramClient

        return AsyncDeepgramClient(api_key=self._api_key())

    @error_logged(re_raise=ProviderError, message="Speech-to-text failed")
    @with_retry()
    async def transcribe(
        self, audio: AudioChunk, config: DeepgramConfig | None = None
    ) -> Transcript:
        config = config or self._default_config()
        client = self._build_client()

        kwargs = dict(
            model=config.model,
            smart_format=config.smart_format,
            punctuate=config.punctuate,
            diarize=config.diarize,
            utterances=True,
        )
        if config.language:
            kwargs["language"] = config.language
        elif config.detect_language:
            kwargs["detect_language"] = True

        response = await client.listen.v1.media.transcribe_file(
            request=audio.data, **kwargs
        )

        results = response.results
        channel = results.channels[0]
        language = parse_language(getattr(results, "language", None) or "en")

        utterances = []
        for alt in channel.alternatives:
            if not alt.words:
                text = alt.paragraphs.transcript if alt.paragraphs else ""
                utterances.append(Utterance(text=text, confidence=alt.confidence))
            else:
                for word in alt.words:
                    speaker = getattr(word, "speaker", None)
                    utterances.append(
                        Utterance(
                            text=word.word,
                            start_ms=int(word.start * 1000),
                            end_ms=int(word.end * 1000),
                            confidence=word.confidence,
                            speaker=str(speaker) if speaker is not None else None,
                        )
                    )

        return Transcript(
            utterances=utterances,
            language=language,
            metadata={"stt_provider": "deepgram", "model": config.model},
        )

    def stream(
        self, frames: AsyncIterator[AudioChunk], config: DeepgramConfig | None = None
    ) -> AsyncIterator[Transcript]:
        config = config or self._default_config()

        async def _stream() -> AsyncIterator[Transcript]:
            client = self._build_client()

            connect_kwargs = dict(
                model=config.model,
                smart_format=config.smart_format,
                punctuate=config.punctuate,
                utterance_end_ms="1000",
            )
            if config.language:
                connect_kwargs["language"] = config.language
            # `detect_language` is not supported for streaming, only pre-recorded audio.

            async with client.listen.v1.connect(**connect_kwargs) as socket:

                async def _send_frames() -> None:
                    async for chunk in frames:
                        await socket.send_media(chunk.data)
                    await socket.send_close_stream()

                sender = asyncio.create_task(_send_frames())
                async for message in socket:
                    if isinstance(message, bytes):
                        continue
                    utterances = _parse_deepgram_result(message.model_dump_json())
                    if utterances:
                        yield Transcript(
                            utterances=utterances,
                            metadata={"stt_provider": "deepgram", "streaming": True},
                        )
                await sender

        return _stream()


def _parse_deepgram_result(raw: str) -> list[Utterance]:
    import json

    data = json.loads(raw)
    channel = data.get("channel", {})
    alternatives = channel.get("alternatives", [])
    if not alternatives:
        return []

    alt = alternatives[0]
    transcript_text = alt.get("transcript", "").strip()
    if not transcript_text:
        return []

    words = alt.get("words", [])
    if words:
        return [
            Utterance(
                text=w["word"],
                start_ms=int(w["start"] * 1000),
                end_ms=int(w["end"] * 1000),
                confidence=w.get("confidence"),
            )
            for w in words
        ]

    return [Utterance(text=transcript_text)]
