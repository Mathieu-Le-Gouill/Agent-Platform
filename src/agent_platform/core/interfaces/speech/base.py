from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from agent_platform.core.interfaces.speech.config import SpeechConfig
from agent_platform.core.schemas.chunk import AudioChunk
from agent_platform.core.schemas.conversation import Transcript

ConfigT = TypeVar("ConfigT", bound=SpeechConfig)


class BaseSpeechToText(ABC, Generic[ConfigT]):
    @abstractmethod
    async def transcribe(
        self, audio: AudioChunk, config: ConfigT | None = None
    ) -> Transcript: ...

    @abstractmethod
    def stream(
        self, frames: AsyncIterator[AudioChunk], config: ConfigT | None = None
    ) -> AsyncIterator[Transcript]: ...
