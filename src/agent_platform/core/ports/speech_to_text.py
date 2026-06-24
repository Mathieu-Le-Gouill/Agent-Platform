from typing import AsyncIterator, Protocol
from core.value_objects.audio_segment import AudioSegment
from core.entities.transcript import Transcript


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