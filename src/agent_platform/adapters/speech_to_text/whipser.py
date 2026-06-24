from platform.core.value_objects.audio_segment import AudioSegment



class WhisperCppClient:
    
    
    async def transcribe(self, audio: AudioSegment) -> str: 
        ...