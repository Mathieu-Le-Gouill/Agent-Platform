from __future__ import annotations

from pydantic import BaseModel, Field

from agent_platform.agents.tools.base import Tool, ToolError
from uuid import uuid4

from agent_platform.agents.tools._utils import safe_call
from agent_platform.integrations.speech.base import BaseSpeechToText
from agent_platform.models.chunk import AudioChunk
from agent_platform.models.conversation import Transcript
from agent_platform.models.enums import AudioFormat, DataType


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

    def __init__(self, provider: BaseSpeechToText) -> None:
        self._provider = provider

    async def run(self, **kwargs) -> Transcript:
        validated = TranscribeInput(**kwargs)
        chunk = AudioChunk(
            id=uuid4(),
            data=validated.data,
            sample_rate=validated.sample_rate,
            channels=validated.channels,
            dtype=DataType.FLOAT32,
            format=AudioFormat.UNKNOWN,
        )
        result = await safe_call(
            self._provider.transcribe(chunk),
            "Speech-to-text failed",
        )
        if not result.utterances:
            raise ToolError("Speech-to-text returned no utterances")
        return result
