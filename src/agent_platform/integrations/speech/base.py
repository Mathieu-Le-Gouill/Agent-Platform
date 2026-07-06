from abc import ABC, abstractmethod
from typing import AsyncIterator

from agent_platform.models.chunk import AudioChunk
from agent_platform.models.conversation import Transcript


class BaseSpeechToText(ABC):
    @abstractmethod
    async def transcribe(self, audio: AudioChunk) -> Transcript: ...

    @abstractmethod
    def stream(self, frames: AsyncIterator[AudioChunk]) -> AsyncIterator[Transcript]: ...