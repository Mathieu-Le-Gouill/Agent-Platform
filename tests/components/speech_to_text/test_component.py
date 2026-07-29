from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from agent_platform.components.speech_to_text.component import SpeechToText
from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.interfaces.speech.config import SpeechConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript, Utterance
from agent_platform.core.schemas.enums import AudioFormat, DataType


class _FakeSpeechToText(BaseSpeechToText):
    def __init__(self) -> None:
        self.calls: list[tuple[AudioChunk, SpeechConfig | None]] = []

    async def transcribe(self, audio, config=None):
        self.calls.append((audio, config))
        return Transcript(utterances=[Utterance(text="hello")])

    async def stream(self, frames, config=None):
        async for frame in frames:
            yield Transcript(utterances=[Utterance(text=f"partial-{frame.id}")])


def _audio_chunk() -> AudioChunk:
    return AudioChunk(
        id=uuid4(),
        data=b"\x00\x01",
        sample_rate=16000,
        channels=1,
        dtype=DataType.FLOAT32,
        format=AudioFormat.WAV,
    )


@pytest.mark.asyncio
async def test_forwards_audio_and_config_to_backend():
    backend = _FakeSpeechToText()
    speech_to_text = SpeechToText(backend)
    chunk = _audio_chunk()
    config = SpeechConfig(model="whisper")

    result = await speech_to_text.arun((chunk, config))

    assert result.utterances[0].text == "hello"
    assert backend.calls[0] == (chunk, config)


@pytest.mark.asyncio
async def test_astream_forwards_frames_to_backend():
    backend = _FakeSpeechToText()
    speech_to_text = SpeechToText(backend)
    chunk = _audio_chunk()

    async def frames() -> AsyncIterator[AudioChunk]:
        yield chunk

    results = [t async for t in speech_to_text.astream(frames())]

    assert results[0].utterances[0].text == f"partial-{chunk.id}"
