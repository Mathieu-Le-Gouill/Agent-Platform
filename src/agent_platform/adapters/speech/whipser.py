from agent_platform.core.entities.audio.audio_segment import AudioSegment



class WhisperCppClient:
    
    
    async def transcribe(self, audio: AudioSegment) -> str: 
        ...