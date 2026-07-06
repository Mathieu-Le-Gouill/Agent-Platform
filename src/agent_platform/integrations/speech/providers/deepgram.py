from __future__ import annotations

import asyncio
from typing import AsyncIterator

from agent_platform.integrations.speech.base import BaseSpeechToText
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.conversation import Transcript, Utterance
from agent_platform.models.enums import Language


class DeepgramSTT(BaseSpeechToText):
    def __init__(self, api_key: str) -> None:
        from deepgram import DeepgramClient

        self._client = DeepgramClient(api_key)

    async def transcribe(self, audio: AudioChunk) -> Transcript:
        from deepgram import PrerecordedOptions

        options = PrerecordedOptions(
            model="nova-2",
            language="en",
            smart_format=True,
            punctuate=True,
            utterances=True,
        )

        payload = {
            "buffer": audio.data,
            "mimetype": _mime_from_format(audio.format.value),
        }

        response = await self._client.listen.asyncprerecorded.v("1").transcribe(  # type: ignore[attr-defined]
            payload, options
        )

        results = response.results
        channels = results.channels[0]
        language = _parse_language(results.get("language") or "en")

        utterances = []
        for alt in channels.alternatives:
            if not alt.words:
                text = alt.paragraphs.transcript if alt.paragraphs else ""
                utterances.append(Utterance(text=text, confidence=alt.confidence))
            else:
                for word in alt.words:
                    utterances.append(
                        Utterance(
                            text=word.word,
                            start_ms=int(word.start * 1000),
                            end_ms=int(word.end * 1000),
                            confidence=word.confidence,
                        )
                    )

        return Transcript(
            utterances=utterances,
            language=language,
            metadata={"stt_provider": "deepgram", "model": "nova-2"},
        )

    async def stream(
        self, frames: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[Transcript]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        dg_live = self._client.listen.asyncwebsocket.v("1")  # type: ignore[attr-defined]

        def _on_result(result: str) -> None:
            queue.put_nowait(result)

        from deepgram import LiveOptions, LiveTranscriptionEvents

        options = LiveOptions(
            model="nova-2",
            language="en",
            smart_format=True,
            punctuate=True,
            utterance_end_ms="1000",
        )

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


def _parse_language(code: str) -> Language | None:
    try:
        return Language(code.split("-")[0].split("_")[0])
    except ValueError:
        return None


def _mime_from_format(fmt: str) -> str:
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "m4a": "audio/mp4",
    }.get(fmt, "audio/wav")
