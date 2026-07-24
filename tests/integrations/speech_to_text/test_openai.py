from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai")

from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import AudioFormat
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.speech_to_text.openai.config import (
    OpenAIWhisperConfig,
)
from agent_platform.integrations.speech_to_text.openai.openai import OpenAIWhisperSTT


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
    async def test_whisper_model_defaults_to_verbose_json(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        with patch.object(stt, "_build_client", return_value=mock_client):
            await stt.transcribe(_make_audio(), OpenAIWhisperConfig(model="whisper-1"))

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["response_format"] == "verbose_json"

    async def test_gpt4o_transcribe_model_defaults_to_json(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        with patch.object(stt, "_build_client", return_value=mock_client):
            await stt.transcribe(
                _make_audio(), OpenAIWhisperConfig(model="gpt-4o-transcribe")
            )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["response_format"] == "json"

    async def test_gpt4o_mini_transcribe_model_defaults_to_json(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        with patch.object(stt, "_build_client", return_value=mock_client):
            await stt.transcribe(
                _make_audio(), OpenAIWhisperConfig(model="gpt-4o-mini-transcribe")
            )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["response_format"] == "json"

    async def test_explicit_response_format_overrides_default(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        with patch.object(stt, "_build_client", return_value=mock_client):
            await stt.transcribe(
                _make_audio(),
                OpenAIWhisperConfig(model="whisper-1", response_format="text"),
            )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["response_format"] == "text"

    async def test_timestamp_granularities_forwarded_only_with_verbose_json(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        with patch.object(stt, "_build_client", return_value=mock_client):
            await stt.transcribe(
                _make_audio(),
                OpenAIWhisperConfig(
                    model="whisper-1", timestamp_granularities=["word"]
                ),
            )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["timestamp_granularities"] == ["word"]

    async def test_timestamp_granularities_dropped_for_json_format(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        with patch.object(stt, "_build_client", return_value=mock_client):
            await stt.transcribe(
                _make_audio(),
                OpenAIWhisperConfig(
                    model="gpt-4o-transcribe", timestamp_granularities=["word"]
                ),
            )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert "timestamp_granularities" not in kwargs

    async def test_prompt_forwarded_when_set(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        with patch.object(stt, "_build_client", return_value=mock_client):
            await stt.transcribe(
                _make_audio(),
                OpenAIWhisperConfig(model="whisper-1", prompt="vocabulary hint"),
            )

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert kwargs["prompt"] == "vocabulary hint"

    async def test_prompt_omitted_when_unset(self):
        stt = OpenAIWhisperSTT(OpenAICredentials(api_key="key"))
        mock_response = MagicMock()
        mock_response.segments = None
        mock_response.text = "hello"
        mock_response.language = "en"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        with patch.object(stt, "_build_client", return_value=mock_client):
            await stt.transcribe(_make_audio(), OpenAIWhisperConfig(model="whisper-1"))

        _, kwargs = mock_client.audio.transcriptions.create.call_args
        assert "prompt" not in kwargs
