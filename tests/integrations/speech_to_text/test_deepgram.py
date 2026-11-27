from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("deepgram")

from agent_platform.core.errors import (
    MissingCredentialError,
    ProviderError,
    ValidationError,
)
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Utterance
from agent_platform.core.schemas.enums import AudioFormat, DataType, Language
from agent_platform.integrations.credentials import DeepgramCredentials
from agent_platform.integrations.speech_to_text.deepgram.config import DeepgramConfig
from agent_platform.integrations.speech_to_text.deepgram.mappers import (
    parse_deepgram_result as _parse_deepgram_result,
)
from agent_platform.integrations.speech_to_text.deepgram.provider import (
    DEFAULT_STT_TIMEOUT_SECONDS,
    DeepgramSTT,
)
from agent_platform.integrations.speech_to_text.language import (
    parse_language as _parse_language,
)


def _make_audio() -> AudioChunk:
    return AudioChunk(data=b"audio_data", start=0, end=1000, format=AudioFormat.WAV)


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

    def test_empty_transcript_text_returns_empty_list(self):
        raw = '{"channel": {"alternatives": [{"transcript": "   ", "words": []}]}}'
        assert _parse_deepgram_result(raw) == []


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


class TestDeepgramApiKey:
    def test_raises_without_api_key(self):
        stt = DeepgramSTT(DeepgramCredentials(api_key=None))
        with pytest.raises(MissingCredentialError):
            stt._api_key()

    def test_returns_key_when_present(self):
        stt = DeepgramSTT(DeepgramCredentials(api_key="secret"))
        assert stt._api_key() == "secret"

    def test_default_config(self):
        stt = DeepgramSTT(DeepgramCredentials(api_key="secret"))
        assert isinstance(stt._default_config(), DeepgramConfig)

    def test_build_client_uses_api_key(self):
        stt = DeepgramSTT(DeepgramCredentials(api_key="secret"))
        client = stt._build_client(DeepgramConfig())
        assert client is not None


class TestDeepgramClientOptions:
    def test_default_timeout_and_sdk_retries_off(self, mocker):
        mock_client_cls = mocker.patch("deepgram.AsyncDeepgramClient", autospec=False)
        stt = DeepgramSTT(DeepgramCredentials(api_key="secret"))
        stt._build_client(DeepgramConfig(max_retries=None))
        _, kwargs = mock_client_cls.call_args
        assert kwargs["max_retries"] == 0
        assert kwargs["timeout"] == DEFAULT_STT_TIMEOUT_SECONDS == 300.0

    def test_explicit_timeout_forwarded_and_sdk_retries_off(self, mocker):
        mock_client_cls = mocker.patch("deepgram.AsyncDeepgramClient", autospec=False)
        stt = DeepgramSTT(DeepgramCredentials(api_key="secret"))
        stt._build_client(DeepgramConfig(timeout=15.0, max_retries=5))
        _, kwargs = mock_client_cls.call_args
        assert kwargs["timeout"] == 15.0
        assert kwargs["max_retries"] == 0

    def test_client_options_fallback(self, mocker):
        from agent_platform.core.credentials import ClientOptions

        mock_client_cls = mocker.patch("deepgram.AsyncDeepgramClient", autospec=False)
        stt = DeepgramSTT(
            DeepgramCredentials(api_key="secret"),
            client_options=ClientOptions(timeout=60.0, max_retries=7),
        )
        stt._build_client(DeepgramConfig())
        _, kwargs = mock_client_cls.call_args
        assert kwargs["timeout"] == 60.0
        assert kwargs["max_retries"] == 0


def _mock_response(channel_json: dict, detected_language: str | None = None):
    alt = MagicMock()
    alt.words = channel_json.get("words")
    alt.paragraphs = (
        MagicMock(transcript=channel_json.get("paragraph_text"))
        if channel_json.get("paragraph_text") is not None
        else None
    )
    alt.confidence = channel_json.get("confidence")
    if alt.words:
        for w in alt.words:
            pass
    channel = MagicMock()
    channel.alternatives = [alt]
    channel.detected_language = detected_language
    results = MagicMock()
    results.channels = [channel]
    response = MagicMock()
    response.results = results
    return response


class TestDeepgramTranscribe:
    async def test_words_mapped_to_utterances(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        word1 = MagicMock(word="hello", start=0.0, end=0.5, confidence=0.9, speaker=1)
        word2 = MagicMock(word="world", start=0.5, end=1.0, confidence=0.8, speaker=1)

        response = _mock_response({"words": [word1, word2]})
        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file = AsyncMock(return_value=response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        transcript = await stt.transcribe(_make_audio(), DeepgramConfig())

        assert len(transcript.utterances) == 2
        assert transcript.utterances[0].text == "hello"
        assert transcript.utterances[0].start_ms == 0
        assert transcript.utterances[0].end_ms == 500
        assert transcript.utterances[0].speaker == "1"

    async def test_no_words_falls_back_to_paragraph_transcript(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        response = _mock_response(
            {"words": None, "paragraph_text": "full text", "confidence": 0.7}
        )
        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file = AsyncMock(return_value=response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        transcript = await stt.transcribe(_make_audio(), DeepgramConfig())

        assert len(transcript.utterances) == 1
        assert transcript.utterances[0].text == "full text"
        assert transcript.utterances[0].confidence == 0.7

    async def test_no_words_no_paragraphs_yields_empty_text(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        response = _mock_response({"words": None, "paragraph_text": None})
        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file = AsyncMock(return_value=response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        transcript = await stt.transcribe(_make_audio(), DeepgramConfig())

        assert transcript.utterances[0].text == ""

    async def test_language_forwarded_when_set(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        response = _mock_response({"words": None, "paragraph_text": "hi"})
        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file = AsyncMock(return_value=response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        await stt.transcribe(_make_audio(), DeepgramConfig(language="fr"))

        _, kwargs = mock_client.listen.v1.media.transcribe_file.call_args
        assert kwargs["language"] == "fr"

    async def test_language_omitted_when_unset(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        response = _mock_response({"words": None, "paragraph_text": "hi"})
        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file = AsyncMock(return_value=response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        await stt.transcribe(_make_audio(), DeepgramConfig())

        _, kwargs = mock_client.listen.v1.media.transcribe_file.call_args
        assert "language" not in kwargs

    @pytest.mark.parametrize(
        ("detected", "configured", "expected"),
        [
            ("fr", None, Language.FR),
            (None, "de", Language.GE),
            (None, None, Language.EN),
        ],
        ids=["detected", "configured", "default"],
    )
    async def test_language_read_from_channel_detection(
        self, mocker, detected, configured, expected
    ):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        response = _mock_response(
            {"words": None, "paragraph_text": "hi"}, detected_language=detected
        )
        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file = AsyncMock(return_value=response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        transcript = await stt.transcribe(
            _make_audio(),
            DeepgramConfig(language=configured, detect_language=configured is None),
        )

        assert transcript.language == expected

    async def test_provider_exception_wrapped(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file = AsyncMock(
            side_effect=RuntimeError("network down")
        )

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        with pytest.raises(ProviderError, match="network down"):
            await stt.transcribe(_make_audio(), DeepgramConfig())


class _FakeSocket:
    def __init__(self, messages):
        self._messages = messages
        self.sent = []
        self.closed = False

    async def send_media(self, data):
        self.sent.append(data)

    async def send_close_stream(self):
        self.closed = True

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for m in self._messages:
            yield m


class _FakeConnectCM:
    def __init__(self, socket):
        self._socket = socket

    async def __aenter__(self):
        return self._socket

    async def __aexit__(self, *exc):
        return False


class TestDeepgramStream:
    async def _frames(self, chunks):
        for c in chunks:
            yield c

    async def test_yields_transcript_per_results_message(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        msg1 = MagicMock()
        msg1.model_dump_json.return_value = (
            '{"channel": {"alternatives": [{"transcript": "hello", "words": []}]}}'
        )
        socket = _FakeSocket([msg1])
        mock_client = MagicMock()
        mock_client.listen.v1.connect = MagicMock(return_value=_FakeConnectCM(socket))

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        results = [
            t async for t in stt.stream(self._frames([_make_audio()]), DeepgramConfig())
        ]

        assert len(results) == 1
        assert results[0].utterances[0].text == "hello"
        assert socket.sent == [b"audio_data"]
        assert socket.closed is True

    async def test_skips_binary_messages(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        socket = _FakeSocket([b"\x00\x01"])
        mock_client = MagicMock()
        mock_client.listen.v1.connect = MagicMock(return_value=_FakeConnectCM(socket))

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        results = [
            t async for t in stt.stream(self._frames([_make_audio()]), DeepgramConfig())
        ]

        assert results == []

    async def test_empty_transcript_yields_nothing(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        msg = MagicMock()
        msg.model_dump_json.return_value = '{"channel": {"alternatives": []}}'
        socket = _FakeSocket([msg])
        mock_client = MagicMock()
        mock_client.listen.v1.connect = MagicMock(return_value=_FakeConnectCM(socket))

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        results = [
            t async for t in stt.stream(self._frames([_make_audio()]), DeepgramConfig())
        ]

        assert results == []

    async def test_language_forwarded_to_connect(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        socket = _FakeSocket([])
        mock_client = MagicMock()
        mock_client.listen.v1.connect = MagicMock(return_value=_FakeConnectCM(socket))

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        results = [
            t
            async for t in stt.stream(
                self._frames([_make_audio()]), DeepgramConfig(language="es")
            )
        ]

        assert results == []
        _, kwargs = mock_client.listen.v1.connect.call_args
        assert kwargs["language"] == "es"

    async def _connect_kwargs(self, mocker, frames):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        socket = _FakeSocket([])
        mock_client = MagicMock()
        mock_client.listen.v1.connect = MagicMock(return_value=_FakeConnectCM(socket))
        mocker.patch.object(stt, "_build_client", return_value=mock_client)

        _ = [t async for t in stt.stream(self._frames(frames), DeepgramConfig())]

        assert socket.sent == [frame.data for frame in frames]
        return mock_client.listen.v1.connect.call_args.kwargs

    @pytest.mark.parametrize(
        ("dtype", "encoding"),
        [(DataType.FLOAT32, "linear32"), (DataType.INT16, "linear16")],
    )
    async def test_raw_pcm_declares_encoding_sample_rate_and_channels(
        self, mocker, dtype, encoding
    ):
        frames = [
            AudioChunk(data=b"\x00\x01" * 8, dtype=dtype, sample_rate=8000),
            AudioChunk(data=b"\x02\x03" * 8, dtype=dtype, sample_rate=8000),
        ]

        kwargs = await self._connect_kwargs(mocker, frames)

        assert kwargs["encoding"] == encoding
        assert kwargs["sample_rate"] == 8000
        assert kwargs["channels"] == 1

    @pytest.mark.parametrize(
        ("data", "audio_format"),
        [
            (b"RIFF\x24\x00\x00\x00WAVEfmt ", AudioFormat.WAV),
            (b"\x00\x00\x00\x20ftypM4A \x00\x00", AudioFormat.M4A),
            (b"ID3\x04\x00\x00\x00\x00\x00\x00", AudioFormat.MP3),
            # MPEG-1 Layer III, 128 kbit/s, 44.1 kHz.
            (b"\xff\xfb\x90\x64\x00\x00", AudioFormat.MP3),
            # ADTS AAC-LC, 44.1 kHz.
            (b"\xff\xf1\x50\x80\x00\x00", AudioFormat.AAC),
        ],
        ids=["wav", "ftyp", "id3", "mpeg-frame", "adts-frame"],
    )
    async def test_containerized_audio_declares_no_encoding(
        self, mocker, data, audio_format
    ):
        frames = [AudioChunk(data=data, format=audio_format)]

        kwargs = await self._connect_kwargs(mocker, frames)

        assert "encoding" not in kwargs
        assert "sample_rate" not in kwargs

    @pytest.mark.parametrize(
        "data",
        [
            b"\xff\xff" * 8,  # int16 -1 repeated: bitrate index 0b1111
            b"\xff\xff\x00\x00" * 4,  # -1 then 0: free-format bitrate
            b"\xff\xfb\x9c\x64" * 4,  # sample-rate index 0b11
            b"\xff\xeb\x90\x64" * 4,  # reserved MPEG version 0b01
        ],
        ids=["bitrate-1111", "bitrate-0000", "srate-11", "version-01"],
    )
    async def test_raw_pcm_with_frame_sync_bytes_keeps_its_encoding(self, mocker, data):
        frames = [AudioChunk(data=data, dtype=DataType.INT16, format=AudioFormat.MP3)]

        kwargs = await self._connect_kwargs(mocker, frames)

        assert kwargs["encoding"] == "linear16"

    async def test_frame_sync_ignored_for_other_formats(self, mocker):
        frames = [
            AudioChunk(
                data=b"\xff\xfb\x90\x64", dtype=DataType.INT16, format=AudioFormat.WAV
            )
        ]

        kwargs = await self._connect_kwargs(mocker, frames)

        assert kwargs["encoding"] == "linear16"

    async def test_empty_stream_opens_no_socket(self, mocker):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        build_client = mocker.patch.object(stt, "_build_client")

        results = [t async for t in stt.stream(self._frames([]), DeepgramConfig())]

        assert results == []
        build_client.assert_not_called()

    @pytest.mark.parametrize(
        "chunk",
        [
            AudioChunk(data=b"\x00\x01", dtype=DataType.INT8),
            AudioChunk(data=b"\x00\x01" * 4, dtype=DataType.INT16, channels=2),
        ],
        ids=["int8", "stereo"],
    )
    async def test_unsupported_raw_audio_is_rejected(self, mocker, chunk):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
        mock_client = MagicMock()
        mocker.patch.object(stt, "_build_client", return_value=mock_client)

        with pytest.raises(ValidationError, match="Deepgram streaming"):
            _ = [t async for t in stt.stream(self._frames([chunk]), DeepgramConfig())]

        mock_client.listen.v1.connect.assert_not_called()
