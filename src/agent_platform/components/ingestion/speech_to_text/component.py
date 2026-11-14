from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from agent_platform.components.base import Component
from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.interfaces.speech.config import SpeechConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript

SpeechConfigT = TypeVar("SpeechConfigT", bound=SpeechConfig)

SpeechToTextInput = tuple[AudioChunk, SpeechConfigT | None]


class SpeechToText(
    Component[SpeechToTextInput[SpeechConfigT], Transcript], Generic[SpeechConfigT]
):
    def __init__(self, backend: BaseSpeechToText[SpeechConfigT]) -> None:
        self._backend = backend

    async def arun(self, input: SpeechToTextInput[SpeechConfigT]) -> Transcript:
        audio, config = input
        return await self._backend.transcribe(audio, config)

    def astream(
        self,
        frames: AsyncIterator[AudioChunk],
        config: SpeechConfigT | None = None,
    ) -> AsyncIterator[Transcript]:
        return self._backend.stream(frames, config)
