from __future__ import annotations

import asyncio
from typing import AsyncIterator

from agent_platform.integrations.credentials import DeepgramCredentials
from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.integrations.speech_to_text.deepgram.config import DeepgramConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.integrations.speech_to_text.utils import parse_language
from agent_platform.core.errors import ProviderError, error_logged, with_retry


class DeepgramSTT(BaseSpeechToText[DeepgramConfig]):
    def __init__(self, credentials: DeepgramCredentials | None = None) -> None:
        self._credentials = (
            credentials if credentials is not None else DeepgramCredentials()
        )

    def _default_config(self) -> DeepgramConfig:
        return DeepgramConfig()

    @error_logged(re_raise=ProviderError, message="Speech-to-text failed")
    @with_retry()
    async def transcribe(
        self, audio: AudioChunk, config: DeepgramConfig | None = None
    ) -> Transcript:
        config = config or self._default_config()
        from deepgram import DeepgramClient, PrerecordedOptions

        client = DeepgramClient(self._credentials.api_key.get_secret_value())

        options_kwargs = dict(
            model=config.model,
            smart_format=config.smart_format,
            punctuate=config.punctuate,
            diarize=config.diarize,
            utterances=True,
        )
        if config.language:
            options_kwargs["language"] = config.language
        options = PrerecordedOptions(**options_kwargs)

        payload = {
            "buffer": audio.data,
            "mimetype": _mime_from_format(audio.format.value),
        }

        response = await client.listen.asyncprerecorded.v("1").transcribe(
            payload, options
        )

        results = response.results
        channels = results.channels[0]
        language = parse_language(getattr(results, "language", None) or "en")

        utterances = []
        for alt in channels.alternatives:
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
            queue: asyncio.Queue[str] = asyncio.Queue()
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

            client = DeepgramClient(self._credentials.api_key.get_secret_value())
            dg_live = client.listen.asyncwebsocket.v("1")

            def _on_result(result: str) -> None:
                queue.put_nowait(result)

            live_kwargs = dict(
                model=config.model,
                smart_format=config.smart_format,
                punctuate=config.punctuate,
                utterance_end_ms="1000",
            )
            if config.language:
                live_kwargs["language"] = config.language
            options = LiveOptions(**live_kwargs)

            dg_live.on(LiveTranscriptionEvents.Transcript, _on_result)
            await dg_live.start(options)

            async def _send_frames() -> None:
                async for chunk in frames:
                    await dg_live.send(chunk.data)
                await dg_live.finish()
                queue.put_nowait("")

            async def _receive_results() -> AsyncIterator[Transcript]:
                while True:
                    raw = await queue.get()
                    if not raw:
                        break
                    utterances = _parse_deepgram_result(raw)
                    if utterances:
                        yield Transcript(
                            utterances=utterances,
                            metadata={"stt_provider": "deepgram", "streaming": True},
                        )

            sender = asyncio.create_task(_send_frames())
            async for transcript in _receive_results():
                yield transcript
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


def _mime_from_format(fmt: str) -> str:
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "m4a": "audio/mp4",
    }.get(fmt, "audio/wav")
