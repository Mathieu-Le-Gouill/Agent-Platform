from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_platform.integrations.speech.providers.deepgram import (
    DeepgramSTT,
    _parse_language,
    _mime_from_format,
    _parse_deepgram_result,
)

try:
    from agent_platform.integrations.speech.providers.whisperx import (
        WhisperXSTT,
    )

    HAS_WHISPERX = True
except ImportError:
    HAS_WHISPERX = False

from agent_platform.integrations.speech.base import BaseSpeechToText
from agent_platform.models.enums import Language, AudioFormat
from agent_platform.models.chunk import AudioChunk


class TestParseLanguage:
    def test_valid_code(self):
        assert _parse_language("en") == Language.EN

    def test_invalid_code_returns_none(self):
        assert _parse_language("zz") is None

    def test_locale_code_strips_region(self):
        assert _parse_language("en-US") == Language.EN

    def test_locale_with_underscore(self):
        assert _parse_language("fr_FR") == Language.FR

    def test_empty_string_returns_none(self):
        assert _parse_language("") is None


class TestMimeFromFormat:
    def test_wav(self):
        assert _mime_from_format("wav") == "audio/wav"

    def test_mp3(self):
        assert _mime_from_format("mp3") == "audio/mpeg"

    def test_flac(self):
        assert _mime_from_format("flac") == "audio/flac"

    def test_ogg(self):
        assert _mime_from_format("ogg") == "audio/ogg"

    def test_m4a(self):
        assert _mime_from_format("m4a") == "audio/mp4"

    def test_unknown_format_falls_back_to_wav(self):
        assert _mime_from_format("unknown") == "audio/wav"

    def test_empty_string_falls_back_to_wav(self):
        assert _mime_from_format("") == "audio/wav"


class TestWhisperXSTTDefaults:
    def test_constructor_defaults(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")
        stt = WhisperXSTT()
        assert stt._model_size == "large-v3"
        assert stt._device == "cpu"
        assert stt._compute_type == "float32"
        assert stt._batch_size == 16
        assert stt._min_duration_ms == 5000
        assert stt._model is None

    def test_constructor_custom(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")
        stt = WhisperXSTT(
            model_size="medium",
            device="cuda",
            compute_type="float16",
            batch_size=8,
            min_duration_ms=3000,
        )
        assert stt._model_size == "medium"
        assert stt._device == "cuda"
        assert stt._compute_type == "float16"
        assert stt._batch_size == 8
        assert stt._min_duration_ms == 3000

    def test_is_speech_to_text(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")
        stt = WhisperXSTT()
        assert isinstance(stt, BaseSpeechToText)


class TestDeepgramSTTDefaults:
    def test_constructor_stores_api_key(self):
        with patch("deepgram.DeepgramClient") as mock_cls:
            stt = DeepgramSTT(api_key="test-key-123")
            mock_cls.assert_called_once_with("test-key-123")
            assert stt._client is mock_cls.return_value

    def test_is_speech_to_text(self):
        with patch("deepgram.DeepgramClient"):
            stt = DeepgramSTT(api_key="key")
            assert isinstance(stt, BaseSpeechToText)


class TestParseDeepgramResult:
    def test_valid_result_with_words(self):
        raw = '{"channel": {"alternatives": [{"transcript": "hello world", "words": [{"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9}, {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.95}]}]}}'
        result = _parse_deepgram_result(raw)
        assert len(result) == 2
        assert result[0].text == "hello"
        assert result[0].start_ms == 0
        assert result[0].end_ms == 500
        assert result[0].confidence == 0.9
        assert result[1].text == "world"
        assert result[1].start_ms == 500
        assert result[1].end_ms == 1000
        assert result[1].confidence == 0.95

    def test_valid_result_without_words(self):
        raw = '{"channel": {"alternatives": [{"transcript": "hello world", "words": []}]}}'
        result = _parse_deepgram_result(raw)
        assert len(result) == 1
        assert result[0].text == "hello world"

    def test_empty_transcript_returns_empty_list(self):
        raw = '{"channel": {"alternatives": [{"transcript": "   ", "words": []}]}}'
        result = _parse_deepgram_result(raw)
        assert result == []

    def test_no_alternatives_returns_empty_list(self):
        raw = '{"channel": {"alternatives": []}}'
        result = _parse_deepgram_result(raw)
        assert result == []

    def test_no_channel_returns_empty_list(self):
        raw = "{}"
        result = _parse_deepgram_result(raw)
        assert result == []

    def test_missing_transcript_key_returns_empty_list(self):
        raw = '{"channel": {"alternatives": [{}]}}'
        result = _parse_deepgram_result(raw)
        assert result == []


class TestWhisperXSTTTranscribe:
    async def test_transcribe_returns_transcript(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {"text": " hello ", "start": 0.0, "end": 1.0, "confidence": 0.95},
                {"text": " world ", "start": 1.0, "end": 2.0, "confidence": 0.90},
            ],
        }

        stt = WhisperXSTT()
        stt._model = mock_model

        audio = AudioChunk(
            data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
        )
        result = await stt.transcribe(audio)

        assert len(result.utterances) == 2
        assert result.utterances[0].text == "hello"
        assert result.utterances[0].start_ms == 0
        assert result.utterances[0].end_ms == 1000
        assert result.utterances[0].confidence == 0.95
        assert result.utterances[1].text == "world"
        assert result.utterances[1].start_ms == 1000
        assert result.utterances[1].end_ms == 2000
        assert result.utterances[1].confidence == 0.90
        assert result.language == Language.EN
        assert result.metadata["stt_provider"] == "whisperx"

    async def test_transcribe_model_load_failure(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        with patch("whisperx.load_model", return_value=None):
            stt = WhisperXSTT()
            audio = AudioChunk(
                data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
            )
            with pytest.raises(RuntimeError, match="Failed to load WhisperX"):
                await stt.transcribe(audio)

    async def test_transcribe_without_segments(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"language": "en", "segments": []}

        stt = WhisperXSTT()
        stt._model = mock_model

        audio = AudioChunk(
            data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
        )
        result = await stt.transcribe(audio)

        assert result.utterances == []


class TestWhisperXSTTStream:
    async def test_stream_yields_transcripts(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {"text": " hello world ", "start": 0.0, "end": 1.0, "confidence": 0.95}
            ],
        }

        async def _frames():
            yield AudioChunk(
                data=b"\x00\x00\x00\x00", start=0, end=3000, format=AudioFormat.WAV
            )
            yield AudioChunk(
                data=b"\x00\x00\x00\x00", start=3000, end=6000, format=AudioFormat.WAV
            )

        stt = WhisperXSTT(min_duration_ms=5000)
        stt._model = mock_model

        results = [t async for t in stt.stream(_frames())]
        assert len(results) == 1
        assert results[0].utterances[0].text == "hello world"

    async def test_stream_with_remaining_buffer(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {"text": " final ", "start": 0.0, "end": 0.5, "confidence": 0.9}
            ],
        }

        async def _frames():
            yield AudioChunk(
                data=b"\x00\x00\x00\x00", start=0, end=2000, format=AudioFormat.WAV
            )

        stt = WhisperXSTT(min_duration_ms=5000)
        stt._model = mock_model

        results = [t async for t in stt.stream(_frames())]
        assert len(results) == 1

    async def test_stream_model_load_failure(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        async def _frames():
            yield AudioChunk(
                data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
            )

        with patch("whisperx.load_model", return_value=None):
            stt = WhisperXSTT(min_duration_ms=100)
            with pytest.raises(RuntimeError, match="Failed to load WhisperX"):
                async for _ in stt.stream(_frames()):
                    pass


class TestDeepgramSTTTranscribe:
    async def test_transcribe_with_words(self):
        with (
            patch("deepgram.DeepgramClient") as mock_dg_cls,
            patch("deepgram.PrerecordedOptions", create=True) as mock_opts_cls,
        ):
            stt = DeepgramSTT(api_key="test-key")

            mock_v1 = AsyncMock()
            stt._client.listen.asyncprerecorded.v.return_value = mock_v1

            mock_word = MagicMock()
            mock_word.word = "hello"
            mock_word.start = 0.0
            mock_word.end = 0.5
            mock_word.confidence = 0.9

            mock_alt = MagicMock()
            mock_alt.words = [mock_word]
            mock_alt.confidence = 0.9
            mock_alt.paragraphs = None

            mock_channel = MagicMock()
            mock_channel.alternatives = [mock_alt]

            mock_response = MagicMock()
            mock_response.results.channels = [mock_channel]
            mock_response.results.get.return_value = "en"
            mock_v1.transcribe = AsyncMock(return_value=mock_response)

            audio = AudioChunk(
                data=b"audio_data", start=0, end=1000, format=AudioFormat.WAV
            )
            result = await stt.transcribe(audio)

            assert len(result.utterances) == 1
            assert result.utterances[0].text == "hello"
            assert result.utterances[0].start_ms == 0
            assert result.utterances[0].end_ms == 500
            assert result.utterances[0].confidence == 0.9
            assert result.language == Language.EN
            assert result.metadata["stt_provider"] == "deepgram"

    async def test_transcribe_without_words_uses_paragraphs(self):
        with (
            patch("deepgram.DeepgramClient") as mock_dg_cls,
            patch("deepgram.PrerecordedOptions", create=True) as mock_opts_cls,
        ):
            stt = DeepgramSTT(api_key="test-key")

            mock_v1 = AsyncMock()
            stt._client.listen.asyncprerecorded.v.return_value = mock_v1

            mock_paragraphs = MagicMock()
            mock_paragraphs.transcript = "full paragraph text"

            mock_alt = MagicMock()
            mock_alt.words = []
            mock_alt.confidence = 0.8
            mock_alt.paragraphs = mock_paragraphs

            mock_channel = MagicMock()
            mock_channel.alternatives = [mock_alt]

            mock_response = MagicMock()
            mock_response.results.channels = [mock_channel]
            mock_response.results.get.return_value = "en"
            mock_v1.transcribe = AsyncMock(return_value=mock_response)

            audio = AudioChunk(
                data=b"audio_data", start=0, end=1000, format=AudioFormat.WAV
            )
            result = await stt.transcribe(audio)

            assert len(result.utterances) == 1
            assert result.utterances[0].text == "full paragraph text"
            assert result.utterances[0].confidence == 0.8
