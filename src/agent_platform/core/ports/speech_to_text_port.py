from typing import Protocol
from agent_platform.core.entities.audio.audio_segment import AudioSegment

class SpeechToTextPort(Protocol):
    async def transcribe(
        self,
        audio: AudioSegment,
    ) -> str: 
        ...