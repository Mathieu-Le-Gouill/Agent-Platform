import asyncio
from collections.abc import AsyncIterator

from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.interfaces.translation.base import BaseTranslator
from agent_platform.core.schemas.chunk import AudioChunk, TextChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.core.schemas.enums import Language


class SpeechTranslationPipeline:
    def __init__(self, stt: BaseSpeechToText, translator: BaseTranslator) -> None:
        self._stt = stt
        self._translator = translator

    async def run(self, audio: AudioChunk, target: Language) -> Transcript:
        transcript = await self._stt.transcribe(audio)
        return await self._translate_transcript(transcript, target)

    def stream(
        self, frames: AsyncIterator[AudioChunk], target: Language
    ) -> AsyncIterator[Transcript]:
        async def _stream() -> AsyncIterator[Transcript]:
            async for transcript in self._stt.stream(frames):
                yield await self._translate_transcript(transcript, target)

        return _stream()

    async def _translate_transcript(
        self, transcript: Transcript, target: Language
    ) -> Transcript:
        if transcript.language == target:
            return transcript

        utterances = await asyncio.gather(
            *[
                self._translate_utterance(utterance, target, transcript.language)
                for utterance in transcript.utterances
            ]
        )
        return transcript.model_copy(update={"utterances": list(utterances)})

    async def _translate_utterance(
        self, utterance: Utterance, target: Language, source: Language | None
    ) -> Utterance:
        if not utterance.text:
            return utterance

        translated = await self._translator.translate(
            TextChunk(text=utterance.text), target=target, source=source
        )
        return utterance.model_copy(update={"text": translated.text})
