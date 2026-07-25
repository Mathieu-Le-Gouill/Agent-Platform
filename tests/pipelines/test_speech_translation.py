from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from agent_platform.core.schemas.chunk import AudioChunk, TextChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.core.schemas.enums import Language
from agent_platform.pipelines.speech_translation import SpeechTranslationPipeline


class TestRun:
    @pytest.fixture
    def audio(self):
        return AudioChunk(data=b"\x00\x00")

    @pytest.fixture
    def mock_translator(self):
        translator = AsyncMock()

        async def _translate(content, target, source=None):
            return TextChunk(text=f"[{target.value}] {content.text}")

        translator.translate = AsyncMock(side_effect=_translate)
        return translator

    async def test_same_language_skips_translation(self, audio, mock_translator):
        stt = AsyncMock()
        stt.transcribe = AsyncMock(
            return_value=Transcript(
                utterances=[Utterance(text="hello")],
                language=Language.EN,
            )
        )
        pipeline = SpeechTranslationPipeline(stt=stt, translator=mock_translator)

        result = await pipeline.run(audio, target=Language.EN)

        assert result.utterances[0].text == "hello"
        mock_translator.translate.assert_not_awaited()

    async def test_different_language_translates_each_utterance(
        self, audio, mock_translator
    ):
        stt = AsyncMock()
        stt.transcribe = AsyncMock(
            return_value=Transcript(
                utterances=[Utterance(text="bonjour"), Utterance(text="salut")],
                language=Language.FR,
            )
        )
        pipeline = SpeechTranslationPipeline(stt=stt, translator=mock_translator)

        result = await pipeline.run(audio, target=Language.EN)

        assert [u.text for u in result.utterances] == [
            "[en] bonjour",
            "[en] salut",
        ]
        assert mock_translator.translate.await_count == 2

    async def test_empty_utterance_text_is_not_translated(self, audio, mock_translator):
        stt = AsyncMock()
        stt.transcribe = AsyncMock(
            return_value=Transcript(
                utterances=[Utterance(text="")],
                language=Language.FR,
            )
        )
        pipeline = SpeechTranslationPipeline(stt=stt, translator=mock_translator)

        result = await pipeline.run(audio, target=Language.EN)

        assert result.utterances[0].text == ""
        mock_translator.translate.assert_not_awaited()

    async def test_unknown_source_language_still_translates(
        self, audio, mock_translator
    ):
        stt = AsyncMock()
        stt.transcribe = AsyncMock(
            return_value=Transcript(
                utterances=[Utterance(text="hola")],
                language=None,
            )
        )
        pipeline = SpeechTranslationPipeline(stt=stt, translator=mock_translator)

        result = await pipeline.run(audio, target=Language.EN)

        assert result.utterances[0].text == "[en] hola"
        mock_translator.translate.assert_awaited_once()
        call = mock_translator.translate.await_args
        assert call.args[0].text == "hola"
        assert call.kwargs == {"target": Language.EN, "source": None}


class TestStream:
    async def test_stream_translates_each_yielded_transcript(self):
        translator = AsyncMock()

        async def _translate(content, target, source=None):
            return TextChunk(text=f"[{target.value}] {content.text}")

        translator.translate = AsyncMock(side_effect=_translate)

        async def _frames() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=b"\x00")

        async def _transcripts(frames):
            yield Transcript(utterances=[Utterance(text="un")], language=Language.FR)
            yield Transcript(utterances=[Utterance(text="deux")], language=Language.FR)

        stt = AsyncMock()
        stt.stream = lambda frames, config=None: _transcripts(frames)
        pipeline = SpeechTranslationPipeline(stt=stt, translator=translator)

        results = [
            transcript
            async for transcript in pipeline.stream(_frames(), target=Language.EN)
        ]

        assert [t.utterances[0].text for t in results] == ["[en] un", "[en] deux"]

    async def test_stream_skips_translation_for_matching_language(self):
        translator = AsyncMock()

        async def _frames() -> AsyncIterator[AudioChunk]:
            yield AudioChunk(data=b"\x00")

        async def _transcripts(frames):
            yield Transcript(utterances=[Utterance(text="hi")], language=Language.EN)

        stt = AsyncMock()
        stt.stream = lambda frames, config=None: _transcripts(frames)
        pipeline = SpeechTranslationPipeline(stt=stt, translator=translator)

        results = [
            transcript
            async for transcript in pipeline.stream(_frames(), target=Language.EN)
        ]

        assert results[0].utterances[0].text == "hi"
        translator.translate.assert_not_awaited()
