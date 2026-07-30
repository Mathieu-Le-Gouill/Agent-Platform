from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("openai")

from agent_platform.core.errors import MissingCredentialError, ProviderError
from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import AudioFormat
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.speech_to_text.openai.config import (
    OpenAIWhisperConfig,
)
from agent_platform.integrations.speech_to_text.openai.provider import OpenAIWhisperSTT


def _make_audio() -> AudioChunk:
    return AudioChunk(data=b"audio_data", start=0, end=1000, format=AudioFormat.WAV)


class TestOpenAIWhisperSTTDefaults:
    def test_is_speech_to_text(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        assert isinstance(stt, BaseSpeechToText)

    def test_default_config(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        cfg = stt._default_config()
        assert cfg.model == "whisper-1"
        assert cfg.response_format is None
        assert cfg.timestamp_granularities is None
        assert cfg.prompt is None


class TestOpenAIWhisperResponseFormatGating:
    async def test_whisper_model_defaults_to_verbose_json(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        await stt.transcribe(_make_audio(), OpenAIWhisperConfig(model="whisper-1"))

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["response_format"] == "verbose_json"

    async def test_gpt4o_transcribe_model_defaults_to_json(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        await stt.transcribe(
            _make_audio(), OpenAIWhisperConfig(model="gpt-4o-transcribe")
        )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["response_format"] == "json"

    async def test_gpt4o_mini_transcribe_model_defaults_to_json(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        await stt.transcribe(
            _make_audio(), OpenAIWhisperConfig(model="gpt-4o-mini-transcribe")
        )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["response_format"] == "json"

    async def test_explicit_response_format_overrides_default(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        await stt.transcribe(
            _make_audio(),
            OpenAIWhisperConfig(model="whisper-1", response_format="text"),
        )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["response_format"] == "text"

    async def test_timestamp_granularities_forwarded_only_with_verbose_json(
        self, mocker
    ):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        await stt.transcribe(
            _make_audio(),
            OpenAIWhisperConfig(model="whisper-1", timestamp_granularities=["word"]),
        )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["timestamp_granularities"] == ["word"]

    async def test_timestamp_granularities_dropped_for_json_format(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        await stt.transcribe(
            _make_audio(),
            OpenAIWhisperConfig(
                model="gpt-4o-transcribe", timestamp_granularities=["word"]
            ),
        )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert "timestamp_granularities" not in kwargs

    async def test_prompt_forwarded_when_set(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        await stt.transcribe(
            _make_audio(),
            OpenAIWhisperConfig(model="whisper-1", prompt="vocabulary hint"),
        )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["prompt"] == "vocabulary hint"

    async def test_prompt_omitted_when_unset(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        await stt.transcribe(_make_audio(), OpenAIWhisperConfig(model="whisper-1"))

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert "prompt" not in kwargs


class TestOpenAIWhisperBuildClient:
    def test_raises_without_api_key(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key=None))
        with pytest.raises(MissingCredentialError):
            stt._build_client(OpenAIWhisperConfig())

    def test_builds_client_with_api_key(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="secret"))
        client = stt._build_client(OpenAIWhisperConfig())
        assert client is not None

    def test_client_kwargs_default_max_retries(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="secret"))
        kwargs = stt._client_kwargs(OpenAIWhisperConfig(max_retries=None))
        assert kwargs["max_retries"] == 3
        assert "timeout" not in kwargs

    def test_client_kwargs_explicit_timeout_and_retries(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="secret"))
        cfg = OpenAIWhisperConfig(timeout=15.0, max_retries=5)
        kwargs = stt._client_kwargs(cfg)
        assert kwargs["timeout"] == 15.0
        assert kwargs["max_retries"] == 5

    def test_client_kwargs_client_options_fallback(self):
        from agent_platform.core.credentials import ClientOptions

        stt = OpenAIWhisperSTT(
            OpenAICredentials(api_key="secret"),
            client_options=ClientOptions(timeout=60.0, max_retries=7),
        )
        kwargs = stt._client_kwargs(OpenAIWhisperConfig())
        assert kwargs["timeout"] == 60.0
        assert kwargs["max_retries"] == 7


class TestOpenAIWhisperTranscribeResponse:
    async def test_segments_mapped_to_utterances(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        seg1 = MagicMock(text=" hello ", start=0.0, end=1.5)
        seg2 = MagicMock(text="world", start=1.5, end=2.0)
        mock_response = MagicMock()
        mock_response.segments = [seg1, seg2]
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        transcript = await stt.transcribe(_make_audio(), OpenAIWhisperConfig())

        assert len(transcript.utterances) == 2
        assert transcript.utterances[0].text == "hello"
        assert transcript.utterances[0].start_ms == 0
        assert transcript.utterances[0].end_ms == 1500
        assert transcript.utterances[1].text == "world"

    async def test_provider_exception_wrapped(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(
            side_effect=RuntimeError("network down")
        )

        mocker.patch.object(stt, "_build_client", return_value=mock_client)
        with pytest.raises(ProviderError, match="network down"):
            await stt.transcribe(_make_audio(), OpenAIWhisperConfig())


class TestOpenAIWhisperStream:
    async def _collect(self, stt, frames, config=None):
        async def gen():
            for chunk in frames:
                yield chunk

        return [t async for t in stt.stream(gen(), config)]

    async def test_buffers_until_min_duration_then_flushes_remainder(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        chunks = [
            _make_audio(),  # duration 1000ms
            AudioChunk(data=b"more", start=1000, end=5200, format=AudioFormat.WAV),
        ]

        transcripts = []

        async def fake_transcribe(audio, config=None):
            transcripts.append(audio)
            from agent_platform.core.schemas.conversation import Transcript, Utterance

            return Transcript(utterances=[Utterance(text="chunk")])

        config = OpenAIWhisperConfig(min_duration_ms=5000)
        mocker.patch.object(stt, "transcribe", side_effect=fake_transcribe)
        results = await self._collect(stt, chunks, config)

        assert len(results) == 1
        assert len(transcripts) == 1
        combined = transcripts[0]
        assert combined.start == 0
        assert combined.end == 5200
        assert combined.data == b"audio_datamore"

    async def test_flushes_leftover_buffer_at_end(self, mocker):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        chunks = [_make_audio()]  # only 1000ms, below default min_duration_ms

        async def fake_transcribe(audio, config=None):
            from agent_platform.core.schemas.conversation import Transcript, Utterance

            return Transcript(utterances=[Utterance(text="leftover")])

        mocker.patch.object(stt, "transcribe", side_effect=fake_transcribe)
        results = await self._collect(stt, chunks, OpenAIWhisperConfig())

        assert len(results) == 1
        assert results[0].utterances[0].text == "leftover"
