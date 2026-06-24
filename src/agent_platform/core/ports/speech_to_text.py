from typing import AsyncIterator, Protocol
from agent_platform.core.entities.audio_segment import AudioSegment
from agent_platform.core.entities.transcript import Transcript


class SpeechToTextPort(Protocol):
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