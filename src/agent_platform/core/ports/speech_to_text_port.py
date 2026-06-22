from typing import Protocol
from core.entities.audio import AudioSegment

class SpeechToTextPort(Protocol):
    async def transcribe(self, audio: AudioSegment) -> str: 
        ...