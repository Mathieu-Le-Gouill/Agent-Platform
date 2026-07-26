import pytest

from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript
from agent_platform.integrations.speech_to_text._base import buffered_stream


def test_base_speech_to_text_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseSpeechToText()  # type: ignore[abstract]


def test_subclass_must_implement_transcribe():
    class MissingTranscribe(BaseSpeechToText):
        async def stream(self, frames):
            yield None
            return

    with pytest.raises(TypeError):
        MissingTranscribe()  # type: ignore[abstract]


def test_subclass_must_implement_stream():
    class MissingStream(BaseSpeechToText):
        async def transcribe(self, audio):
            return None

    with pytest.raises(TypeError):
        MissingStream()  # type: ignore[abstract]


def test_concrete_subclass_instantiates():
    class Concrete(BaseSpeechToText):
        async def transcribe(self, audio):
            return None

        async def stream(self, frames):
            yield None
            return

    instance = Concrete()
    assert isinstance(instance, BaseSpeechToText)


def _chunk(start: int, end: int) -> AudioChunk:
    return AudioChunk(start=start, end=end)


async def _frames(*chunks: AudioChunk):
    for chunk in chunks:
        yield chunk


async def test_buffered_stream_flushes_once_min_duration_reached():
    seen: list[list[AudioChunk]] = []

    async def transcribe(buffer: list[AudioChunk]) -> Transcript:
        seen.append(buffer)
        return Transcript(utterances=[], metadata={"n": len(buffer)})

    chunks = [_chunk(0, 600), _chunk(600, 1100)]
    results = [t async for t in buffered_stream(_frames(*chunks), 1000, transcribe)]

    assert len(results) == 1
    assert seen == [chunks]
    assert results[0].metadata == {"n": 2}


async def test_buffered_stream_flushes_remainder_below_threshold():
    seen: list[list[AudioChunk]] = []

    async def transcribe(buffer: list[AudioChunk]) -> Transcript:
        seen.append(buffer)
        return Transcript(utterances=[])

    chunk = _chunk(0, 500)
    results = [t async for t in buffered_stream(_frames(chunk), 1000, transcribe)]

    assert len(results) == 1
    assert seen == [[chunk]]


async def test_buffered_stream_yields_nothing_for_empty_input():
    async def transcribe(buffer: list[AudioChunk]) -> Transcript:
        raise AssertionError("should not be called")

    results = [t async for t in buffered_stream(_frames(), 1000, transcribe)]

    assert results == []


async def test_buffered_stream_resets_buffer_between_windows():
    seen: list[list[AudioChunk]] = []

    async def transcribe(buffer: list[AudioChunk]) -> Transcript:
        seen.append(buffer)
        return Transcript(utterances=[])

    first_window = [_chunk(0, 1000)]
    second_window = [_chunk(1000, 2000)]
    results = [
        t
        async for t in buffered_stream(
            _frames(*first_window, *second_window), 1000, transcribe
        )
    ]

    assert len(results) == 2
    assert seen == [first_window, second_window]
