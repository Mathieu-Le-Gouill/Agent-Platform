from typing import AsyncIterator, Protocol
from agent_platform.models.audio import AudioSegment
from agent_platform.models.transcript import Transcript


class BaseSpeechToText(Protocol):
    async def transcribe(
        self, 
        audio_segment: AudioSegment
    ) -> Transcript: 
        ...


    def stream(
        self,
        frames: AsyncIterator[AudioSegment]
    ) -> AsyncIterator[Transcript]: 
        ...