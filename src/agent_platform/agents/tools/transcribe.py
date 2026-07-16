from __future__ import annotations

from typing import AsyncIterator

from pydantic import BaseModel, Field

from agent_platform.agents.tools.base import Tool, ToolError
from uuid import uuid4

from agent_platform.agents.tools._utils import safe_call
from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript
from agent_platform.core.schemas.enums import AudioFormat, DataType


class TranscribeInput(BaseModel):
    data: bytes = Field(..., description="Raw audio bytes")
    sample_rate: int = Field(
        default=16000, ge=8000, le=192000, description="Sample rate in Hz"
    )
    channels: int = Field(default=1, ge=1, le=8, description="Number of audio channels")


class TranscribeTool(Tool):
    name = "transcribe"
    description = "Transcribe audio content to text using speech-to-text."
    input_schema = TranscribeInput
    output_schema = Transcript
    supports_streaming = True

    def __init__(self, provider: BaseSpeechToText) -> None:
        self._provider = provider

    def _build_chunk(self, validated: TranscribeInput) -> AudioChunk:
        return AudioChunk(
            id=uuid4(),
            data=validated.data,
            sample_rate=validated.sample_rate,
            channels=validated.channels,
            dtype=DataType.FLOAT32,
            format=AudioFormat.UNKNOWN,
        )

    async def run(self, **kwargs) -> Transcript:
        validated = TranscribeInput(**kwargs)
        result = await safe_call(
            self._provider.transcribe(self._build_chunk(validated)),
            "Speech-to-text failed",
        )
        if not result.utterances:
            raise ToolError("Speech-to-text returned no utterances")
        return result

    async def astream(self, **kwargs) -> AsyncIterator[str]:
        validated = TranscribeInput(**kwargs)

        async def frames() -> AsyncIterator[AudioChunk]:
            yield self._build_chunk(validated)

        try:
            async for partial in self._provider.stream(frames()):
                for utterance in partial.utterances:
                    if utterance.text:
                        yield utterance.text
        except Exception as exc:
            raise ToolError(f"Speech-to-text streaming failed: {exc}") from exc
