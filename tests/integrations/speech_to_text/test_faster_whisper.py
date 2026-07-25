import threading
from unittest.mock import MagicMock

import pytest

pytest.importorskip("faster_whisper")

from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.enums import AudioFormat
from agent_platform.integrations.speech_to_text.faster_whisper.config import (
    FasterWhisperConfig,
)
from agent_platform.integrations.speech_to_text.faster_whisper.provider import (
    FasterWhisperSTT,
)


def _make_audio() -> AudioChunk:
    import numpy as np

    data = np.zeros(16, dtype=np.float32).tobytes()
    return AudioChunk(data=data, start=0, end=1000, format=AudioFormat.WAV)


def _make_segment(text: str, start: float, end: float, words=None):
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    seg.words = words
    return seg


def _make_word(word: str, start: float, end: float, probability: float):
    w = MagicMock()
    w.word = word
    w.start = start
    w.end = end
    w.probability = probability
    return w


def _make_info(language: str = "en"):
    info = MagicMock()
    info.language = language
    return info


class TestFasterWhisperSTTDefaults:
    def test_is_speech_to_text(self):
        stt = FasterWhisperSTT()
        assert isinstance(stt, BaseSpeechToText)

    def test_default_config(self):
        stt = FasterWhisperSTT()
        cfg = stt._default_config()
        assert cfg.word_timestamps is False
        assert cfg.condition_on_previous_text is True


class TestFasterWhisperTranscribeThreadOffload:
    async def test_transcribe_materializes_segments_inside_thread(self, mocker):
        stt = FasterWhisperSTT()
        mock_model = MagicMock()
        segments = [_make_segment("hello", 0.0, 0.5)]
        mock_model.transcribe.return_value = (iter(segments), _make_info())

        calling_thread: dict = {}

        real_to_thread_target = mock_model.transcribe

        def _tracking_transcribe(*args, **kwargs):
            calling_thread["thread"] = threading.current_thread().ident
            return real_to_thread_target(*args, **kwargs)

        mock_model.transcribe = _tracking_transcribe

        mocker.patch.object(stt, "_get_model", return_value=mock_model)
        main_thread = threading.current_thread().ident
        result = await stt.transcribe(_make_audio())

        assert calling_thread["thread"] != main_thread
        assert len(result.utterances) == 1
        assert result.utterances[0].text == "hello"

    async def test_transcribe_returns_expected_utterances(self, mocker):
        stt = FasterWhisperSTT()
        mock_model = MagicMock()
        segments = [
            _make_segment("hello", 0.0, 0.5),
            _make_segment("world", 0.5, 1.0),
        ]
        mock_model.transcribe.return_value = (iter(segments), _make_info())

        mocker.patch.object(stt, "_get_model", return_value=mock_model)
        result = await stt.transcribe(_make_audio())

        assert [u.text for u in result.utterances] == ["hello", "world"]
        assert result.metadata["stt_provider"] == "faster-whisper"


class TestFasterWhisperWordTimestamps:
    async def test_word_timestamps_disabled_by_default(self, mocker):
        stt = FasterWhisperSTT()
        mock_model = MagicMock()
        words = [_make_word("hello", 0.0, 0.5, 0.9)]
        segments = [_make_segment("hello", 0.0, 0.5, words=words)]
        mock_model.transcribe.return_value = (iter(segments), _make_info())

        mocker.patch.object(stt, "_get_model", return_value=mock_model)
        result = await stt.transcribe(_make_audio())

        assert len(result.utterances) == 1
        assert result.utterances[0].text == "hello"

    async def test_word_timestamps_enabled_expands_to_per_word_utterances(self, mocker):
        stt = FasterWhisperSTT()
        mock_model = MagicMock()
        words = [
            _make_word("hello", 0.0, 0.3, 0.9),
            _make_word("world", 0.3, 0.6, 0.85),
        ]
        segments = [_make_segment("hello world", 0.0, 0.6, words=words)]
        mock_model.transcribe.return_value = (iter(segments), _make_info())

        mocker.patch.object(stt, "_get_model", return_value=mock_model)
        result = await stt.transcribe(
            _make_audio(), FasterWhisperConfig(word_timestamps=True)
        )

        assert len(result.utterances) == 2
        assert result.utterances[0].text == "hello"
        assert result.utterances[0].start_ms == 0
        assert result.utterances[0].end_ms == 300
        assert result.utterances[0].confidence == 0.9
        assert result.utterances[1].text == "world"
        assert result.utterances[1].confidence == 0.85

    async def test_word_timestamps_enabled_forwarded_to_model(self, mocker):
        stt = FasterWhisperSTT()
        mock_model = MagicMock()
        segments = [_make_segment("hello", 0.0, 0.5, words=None)]
        mock_model.transcribe.return_value = (iter(segments), _make_info())

        mocker.patch.object(stt, "_get_model", return_value=mock_model)
        await stt.transcribe(_make_audio(), FasterWhisperConfig(word_timestamps=True))

        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["word_timestamps"] is True

    async def test_condition_on_previous_text_forwarded(self, mocker):
        stt = FasterWhisperSTT()
        mock_model = MagicMock()
        segments = [_make_segment("hello", 0.0, 0.5)]
        mock_model.transcribe.return_value = (iter(segments), _make_info())

        mocker.patch.object(stt, "_get_model", return_value=mock_model)
        await stt.transcribe(
            _make_audio(),
            FasterWhisperConfig(condition_on_previous_text=False),
        )

        _, kwargs = mock_model.transcribe.call_args
        assert kwargs["condition_on_previous_text"] is False


class TestFasterWhisperStream:
    async def test_stream_yields_transcripts(self, mocker):
        stt = FasterWhisperSTT()
        mock_model = MagicMock()
        segments = [_make_segment("hello world", 0.0, 1.0)]
        mock_model.transcribe.return_value = (iter(segments), _make_info())

        async def _frames():
            yield _make_audio()

        mocker.patch.object(stt, "_get_model", return_value=mock_model)
        results = [
            t
            async for t in stt.stream(_frames(), FasterWhisperConfig(min_duration_ms=0))
        ]

        assert len(results) == 1
        assert results[0].utterances[0].text == "hello world"
