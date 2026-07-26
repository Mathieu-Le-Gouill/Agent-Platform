from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_platform.agents.tools._utils import safe_call, safe_stream
from agent_platform.agents.tools.base import Tool, ToolError
from agent_platform.components.speech_to_text import SpeechToText
from agent_platform.core.interfaces.speech.config import SpeechConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript
from agent_platform.core.schemas.document import AudioDocument
from agent_platform.core.schemas.enums import AudioFormat, DataType
from agent_platform.core.schemas.message import AudioBlock, ContentBlock, TextBlock


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

    def __init__(
        self,
        speech_to_text: SpeechToText,
        default_config: SpeechConfig | None = None,
    ) -> None:
        self._speech_to_text = speech_to_text
        self._default_config = default_config

    def _build_chunk(self, validated: TranscribeInput) -> AudioChunk:
        return AudioChunk(
            id=uuid4(),
            data=validated.data,
            sample_rate=validated.sample_rate,
            channels=validated.channels,
            dtype=DataType.FLOAT32,
            format=AudioFormat.UNKNOWN,
        )

    async def run(self, **kwargs: Any) -> Transcript:
        validated = TranscribeInput(**kwargs)
        result = await safe_call(
            self._speech_to_text.arun(
                (self._build_chunk(validated), self._default_config)
            ),
            "Speech-to-text failed",
        )
        if not result.utterances:
            raise ToolError("Speech-to-text returned no utterances")
        return result

    def to_blocks(
        self, chunk: AudioChunk, transcript: Transcript
    ) -> list[ContentBlock]:
        blocks: list[ContentBlock] = [
            AudioBlock(
                audio=AudioDocument(
                    content=chunk.data,
                    format=chunk.format,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                )
            )
        ]
        text = " ".join(u.text for u in transcript.utterances if u.text)
        if text:
            blocks.append(TextBlock(text=text))
        return blocks

    async def astream(self, **kwargs: Any) -> AsyncIterator[str]:
        validated = TranscribeInput(**kwargs)

        async def frames() -> AsyncIterator[AudioChunk]:
            yield self._build_chunk(validated)

        async for partial in safe_stream(
            self._speech_to_text.astream(frames(), config=self._default_config),
            "Speech-to-text streaming failed",
        ):
            for utterance in partial.utterances:
                if utterance.text:
                    yield utterance.text
