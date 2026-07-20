from uuid import uuid4

import pytest

from agent_platform.integrations.speech_to_text.deepgram.deepgram import (
    _parse_deepgram_result,
    _mime_from_format,
)
from agent_platform.integrations.speech_to_text.utils import (
    parse_language as _parse_language,
)

from agent_platform.core.schemas.conversation import Utterance
from agent_platform.core.schemas.enums import Language


class TestParseDeepgramResult:
    def test_returns_list_of_utterances(self):
        raw = '{"channel": {"alternatives": [{"transcript": "hello world", "words": [{"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9}]}]}}'
        result = _parse_deepgram_result(raw)
        assert len(result) == 1
        assert isinstance(result[0], Utterance)

    def test_all_word_fields_mapped_correctly(self):
        raw = '{"channel": {"alternatives": [{"transcript": "test sentence", "words": [{"word": "test", "start": 0.1, "end": 0.3, "confidence": 0.85}]}]}}'
        result = _parse_deepgram_result(raw)
        u = result[0]
        assert u.text == "test"
        assert u.start_ms == 100
        assert u.end_ms == 300
        assert u.confidence == 0.85

    def test_returns_transcript_text_when_no_words(self):
        raw = (
            '{"channel": {"alternatives": [{"transcript": "plain text", "words": []}]}}'
        )
        result = _parse_deepgram_result(raw)
        assert len(result) == 1
        assert result[0].text == "plain text"

    def test_handles_confidence_none(self):
        raw = '{"channel": {"alternatives": [{"transcript": "hi", "words": [{"word": "hi", "start": 0.0, "end": 0.2}]}]}}'
        result = _parse_deepgram_result(raw)
        assert result[0].confidence is None

    def test_strips_transcript_whitespace(self):
        raw = '{"channel": {"alternatives": [{"transcript": "   spaced out   ", "words": []}]}}'
        result = _parse_deepgram_result(raw)
        assert result[0].text == "spaced out"

    def test_handles_large_integer_timestamps(self):
        raw = '{"channel": {"alternatives": [{"transcript": "long", "words": [{"word": "long", "start": 1234.567, "end": 5678.901, "confidence": 0.5}]}]}}'
        result = _parse_deepgram_result(raw)
        assert result[0].start_ms == 1234567
        assert result[0].end_ms == 5678901

    def test_handles_zero_timestamps(self):
        raw = '{"channel": {"alternatives": [{"transcript": "start", "words": [{"word": "start", "start": 0.0, "end": 0.0, "confidence": 1.0}]}]}}'
        result = _parse_deepgram_result(raw)
        assert result[0].start_ms == 0
        assert result[0].end_ms == 0

    def test_multiple_words(self):
        raw = '{"channel": {"alternatives": [{"transcript": "a b c", "words": [{"word": "a", "start": 0.0, "end": 0.1, "confidence": 0.9}, {"word": "b", "start": 0.1, "end": 0.2, "confidence": 0.8}, {"word": "c", "start": 0.2, "end": 0.3, "confidence": 0.7}]}]}}'
        result = _parse_deepgram_result(raw)
        assert len(result) == 3
        assert [u.text for u in result] == ["a", "b", "c"]

    def test_handles_malformed_json(self):
        with pytest.raises(Exception):
            _parse_deepgram_result("not json")


class TestParseLanguage:
    def test_valid_language_code(self):
        assert _parse_language("fr") == Language.FR

    def test_german_locale(self):
        assert _parse_language("de") == Language.GE

    def test_chinese_locale(self):
        assert _parse_language("zh-CN") == Language.CH

    def test_japanese_locale(self):
        assert _parse_language("ja-JP") == Language.JA

    def test_korean_locale(self):
        assert _parse_language("ko-KR") == Language.KO

    def test_spanish_locale_with_underscore(self):
        assert _parse_language("es_MX") == Language.SP

    def test_portuguese_brazil(self):
        assert _parse_language("pt-BR") == Language.PO

    def test_russian(self):
        assert _parse_language("ru") == Language.RU

    def test_italian(self):
        assert _parse_language("it") == Language.IT

    def test_amharic_code(self):
        assert _parse_language("am") == Language.AM

    def test_azerbaijani_code(self):
        assert _parse_language("az") == Language.AZ

    def test_assamese_code(self):
        assert _parse_language("as") == Language.AS

    def test_afrikaans_code(self):
        assert _parse_language("af") == Language.AF

    def test_arabic_code(self):
        assert _parse_language("ar") == Language.AR

    def test_invalid_code_returns_none(self):
        assert _parse_language("invalid") is None

    def test_numbers_code_returns_none(self):
        assert _parse_language("123") is None

    def test_empty_string_returns_none(self):
        assert _parse_language("") is None

    def test_whitespace_code_returns_none(self):
        assert _parse_language("   ") is None


class TestMimeFromFormat:
    def test_wav_format(self):
        assert _mime_from_format("wav") == "audio/wav"

    def test_mp3_format(self):
        assert _mime_from_format("mp3") == "audio/mpeg"

    def test_flac_format(self):
        assert _mime_from_format("flac") == "audio/flac"

    def test_ogg_format(self):
        assert _mime_from_format("ogg") == "audio/ogg"

    def test_m4a_format(self):
        assert _mime_from_format("m4a") == "audio/mp4"

    def test_unknown_format_fallback(self):
        assert _mime_from_format("aiff") == "audio/wav"

    def test_none_format_fallback(self):
        assert _mime_from_format(None) == "audio/wav"

    def test_case_sensitive(self):
        assert _mime_from_format("WAV") == "audio/wav"

    def test_empty_format_fallback(self):
        assert _mime_from_format("") == "audio/wav"
