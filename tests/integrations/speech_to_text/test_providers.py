from unittest.mock import MagicMock

import pytest

pytest.importorskip("deepgram")

from agent_platform.integrations.credentials import DeepgramCredentials
from agent_platform.integrations.speech_to_text.deepgram.provider import (
    DeepgramSTT,
    _parse_deepgram_result,
)
from agent_platform.integrations.speech_to_text.utils import (
    parse_language as _parse_language,
)
from agent_platform.integrations.speech_to_text.whisperx.config import WhisperXConfig

try:
    from agent_platform.integrations.speech_to_text.whisperx.provider import (
        WhisperXSTT,
    )

    HAS_WHISPERX = True
except ImportError:
    HAS_WHISPERX = False

from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import AudioFormat, Language


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


class TestWhisperXSTTDefaults:
    def test_constructor_default_config(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")
        stt = WhisperXSTT()
        cfg = stt._default_config()
        assert cfg.model_size == "large-v3"
        assert cfg.device == "cpu"
        assert cfg.compute_type == "float32"
        assert cfg.batch_size == 16
        assert cfg.min_duration_ms == 5000

    def test_constructor_custom_config_via_default(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")
        stt = WhisperXSTT()
        assert isinstance(stt, BaseSpeechToText)

    def test_is_speech_to_text(self):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")
        stt = WhisperXSTT()
        assert isinstance(stt, BaseSpeechToText)


class TestDeepgramSTTDefaults:
    def test_constructor_stores_credentials(self):
        stt = DeepgramSTT(DeepgramCredentials(api_key="test-key-123"))
        assert stt._credentials.api_key.get_secret_value() == "test-key-123"

    def test_is_speech_to_text(self):
        stt = DeepgramSTT(DeepgramCredentials(api_key="key"))
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
    async def test_transcribe_returns_transcript(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {"text": " hello ", "start": 0.0, "end": 1.0},
                {"text": " world ", "start": 1.0, "end": 2.0},
            ],
        }

        stt = WhisperXSTT()
        audio = AudioChunk(
            data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
        )
        mocker.patch.object(stt, "_load_model", return_value=mock_model)
        result = await stt.transcribe(audio, WhisperXConfig(align=False))

        assert len(result.utterances) == 2
        assert result.utterances[0].text == "hello"
        assert result.utterances[0].start_ms == 0
        assert result.utterances[0].end_ms == 1000
        assert result.utterances[1].text == "world"
        assert result.utterances[1].start_ms == 1000
        assert result.utterances[1].end_ms == 2000
        assert result.language == Language.EN
        assert result.metadata["stt_provider"] == "whisperx"

    async def test_transcribe_without_segments(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"language": "en", "segments": []}

        stt = WhisperXSTT()
        audio = AudioChunk(
            data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
        )
        mocker.patch.object(stt, "_load_model", return_value=mock_model)
        result = await stt.transcribe(audio, WhisperXConfig(align=False))

        assert result.utterances == []


class TestWhisperXSTTStream:
    async def test_stream_yields_transcripts(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"text": " hello world ", "start": 0.0, "end": 1.0}],
        }

        async def _frames():
            yield AudioChunk(
                data=b"\x00\x00\x00\x00", start=0, end=3000, format=AudioFormat.WAV
            )
            yield AudioChunk(
                data=b"\x00\x00\x00\x00", start=3000, end=6000, format=AudioFormat.WAV
            )

        stt = WhisperXSTT()
        mocker.patch.object(stt, "_load_model", return_value=mock_model)
        results = [
            t
            async for t in stt.stream(
                _frames(), WhisperXConfig(min_duration_ms=5000, align=False)
            )
        ]
        assert len(results) == 1
        assert results[0].utterances[0].text == "hello world"

    async def test_stream_with_remaining_buffer(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"text": " final ", "start": 0.0, "end": 0.5}],
        }

        async def _frames():
            yield AudioChunk(
                data=b"\x00\x00\x00\x00", start=0, end=2000, format=AudioFormat.WAV
            )

        stt = WhisperXSTT()
        mocker.patch.object(stt, "_load_model", return_value=mock_model)
        results = [
            t
            async for t in stt.stream(
                _frames(), WhisperXConfig(min_duration_ms=5000, align=False)
            )
        ]
        assert len(results) == 1


class TestWhisperXModelCaching:
    def test_load_model_called_once_across_multiple_transcribe_calls(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        import asyncio

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"text": " hi ", "start": 0.0, "end": 0.5}],
        }

        stt = WhisperXSTT()
        audio = AudioChunk(
            data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
        )

        mock_load = mocker.patch.object(
            stt, "_load_model_sync", return_value=mock_model
        )
        asyncio.run(stt.transcribe(audio, WhisperXConfig(align=False)))
        asyncio.run(stt.transcribe(audio, WhisperXConfig(align=False)))
        asyncio.run(stt.transcribe(audio, WhisperXConfig(align=False)))

        assert mock_load.call_count == 1

    def test_load_model_sync_calls_whisperx_load_model(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        stt = WhisperXSTT()
        mock_load_model = mocker.patch(
            "agent_platform.integrations.speech_to_text.whisperx.provider.whisperx.load_model",
            return_value="the-model",
        )
        result = stt._load_model_sync(
            WhisperXConfig(model_size="tiny", device="cpu", compute_type="int8")
        )

        assert result == "the-model"
        mock_load_model.assert_called_once_with(
            "tiny", device="cpu", compute_type="int8"
        )

    async def test_transcribe_reuses_cached_model_instance(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"text": " hi ", "start": 0.0, "end": 0.5}],
        }

        stt = WhisperXSTT()
        stt._model = mock_model
        audio = AudioChunk(
            data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
        )

        mock_load_sync = mocker.patch.object(stt, "_load_model_sync")
        await stt.transcribe(audio, WhisperXConfig(align=False))

        mock_load_sync.assert_not_called()


class TestWhisperXAlignment:
    async def test_align_called_when_enabled(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"text": " hi ", "start": 0.0, "end": 0.5}],
        }

        aligned_result = {
            "segments": [
                {
                    "text": " hi ",
                    "start": 0.0,
                    "end": 0.5,
                    "words": [{"word": "hi", "start": 0.0, "end": 0.5, "score": 0.8}],
                }
            ]
        }

        stt = WhisperXSTT()
        audio = AudioChunk(
            data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
        )

        mocker.patch.object(stt, "_load_model", return_value=mock_model)
        mock_align = mocker.patch.object(stt, "_align", return_value=aligned_result)
        result = await stt.transcribe(audio, WhisperXConfig(align=True))

        mock_align.assert_called_once()
        assert result.utterances[0].confidence == 0.8

    async def test_align_skipped_when_disabled(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"text": " hi ", "start": 0.0, "end": 0.5}],
        }

        stt = WhisperXSTT()
        audio = AudioChunk(
            data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
        )

        mocker.patch.object(stt, "_load_model", return_value=mock_model)
        mock_align = mocker.patch.object(stt, "_align")
        await stt.transcribe(audio, WhisperXConfig(align=False))

        mock_align.assert_not_called()

    def test_align_model_cached_per_language(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        import asyncio

        stt = WhisperXSTT()
        mock_load_align = mocker.patch(
            "agent_platform.integrations.speech_to_text.whisperx.provider.whisperx.load_align_model",
            return_value=("model_a", {"lang": "en"}),
        )
        asyncio.run(stt._ensure_align_model("en", "cpu"))
        asyncio.run(stt._ensure_align_model("en", "cpu"))

        assert mock_load_align.call_count == 1

    async def test_align_calls_whisperx_align(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        import numpy as np

        stt = WhisperXSTT()
        result = {"segments": [{"text": "hi", "start": 0.0, "end": 0.5}]}
        audio_np = np.zeros((1, 16000), dtype=np.float32)

        mocker.patch.object(
            stt,
            "_ensure_align_model",
            return_value=("model_a", {"lang": "en"}),
        )
        mock_align = mocker.patch(
            "agent_platform.integrations.speech_to_text.whisperx.provider.whisperx.align",
            return_value={"segments": [{"text": "hi aligned"}]},
        )
        aligned = await stt._align(result, audio_np, "en", WhisperXConfig())

        mock_align.assert_called_once()
        assert aligned["segments"][0]["text"] == "hi aligned"


class TestWhisperXDiarization:
    async def test_diarize_called_when_enabled(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"text": " hi ", "start": 0.0, "end": 0.5}],
        }

        diarized_result = {
            "segments": [
                {"text": " hi ", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}
            ]
        }

        stt = WhisperXSTT()
        audio = AudioChunk(
            data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
        )

        mocker.patch.object(stt, "_load_model", return_value=mock_model)
        mock_diarize = mocker.patch.object(
            stt, "_diarize", return_value=diarized_result
        )
        result = await stt.transcribe(audio, WhisperXConfig(align=False, diarize=True))

        mock_diarize.assert_called_once()
        assert result.utterances[0].speaker == "SPEAKER_00"

    async def test_diarize_skipped_by_default(self, mocker):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": [{"text": " hi ", "start": 0.0, "end": 0.5}],
        }

        stt = WhisperXSTT()
        audio = AudioChunk(
            data=b"\x00\x00\x00\x00", start=0, end=1000, format=AudioFormat.WAV
        )

        mocker.patch.object(stt, "_load_model", return_value=mock_model)
        mock_diarize = mocker.patch.object(stt, "_diarize")
        await stt.transcribe(audio, WhisperXConfig(align=False))

        mock_diarize.assert_not_called()

    async def test_diarize_calls_pipeline_and_assigns_speakers(self, monkeypatch):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        import sys
        import types

        import numpy as np

        fake_pipeline_instance = MagicMock(return_value="diarize_df")
        fake_pipeline_cls = MagicMock(return_value=fake_pipeline_instance)
        fake_assign = MagicMock(return_value={"segments": ["assigned"]})

        fake_module = types.ModuleType("whisperx.diarize")
        fake_module.DiarizationPipeline = fake_pipeline_cls
        fake_module.assign_word_speakers = fake_assign
        monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_module)

        stt = WhisperXSTT()
        audio_np = np.zeros((1, 16000), dtype=np.float32)
        config = WhisperXConfig(min_speakers=1, max_speakers=2, hf_token="tok")

        result = await stt._diarize(audio_np, {"segments": []}, config)

        fake_pipeline_cls.assert_called_once_with(token="tok", device="cpu")
        fake_pipeline_instance.assert_called_once_with(
            audio_np, min_speakers=1, max_speakers=2
        )
        fake_assign.assert_called_once_with("diarize_df", {"segments": []})
        assert result == {"segments": ["assigned"]}

    async def test_diarize_pipeline_cached_across_calls(self, monkeypatch):
        if not HAS_WHISPERX:
            pytest.skip("whisperx not available")

        import sys
        import types

        import numpy as np

        fake_pipeline_instance = MagicMock(return_value="diarize_df")
        fake_pipeline_cls = MagicMock(return_value=fake_pipeline_instance)
        fake_assign = MagicMock(return_value={"segments": []})

        fake_module = types.ModuleType("whisperx.diarize")
        fake_module.DiarizationPipeline = fake_pipeline_cls
        fake_module.assign_word_speakers = fake_assign
        monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_module)

        stt = WhisperXSTT()
        audio_np = np.zeros((1, 16000), dtype=np.float32)
        config = WhisperXConfig()

        await stt._diarize(audio_np, {"segments": []}, config)
        await stt._diarize(audio_np, {"segments": []}, config)

        assert fake_pipeline_cls.call_count == 1
