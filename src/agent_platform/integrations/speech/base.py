from abc import ABC, abstractmethod
from typing import AsyncIterator

from agent_platform.models.audio import AudioSegment
from agent_platform.models.transcript import Transcript


class BaseSpeechToText(ABC):
    @abstractmethod
    async def transcribe(
        self, 
        audio_segment: AudioSegment
    ) -> Transcript: 
        ...

    @abstractmethod
    def stream(
        self,
        frames: AsyncIterator[AudioSegment]
    ) -> AsyncIterator[Transcript]: 
        ...