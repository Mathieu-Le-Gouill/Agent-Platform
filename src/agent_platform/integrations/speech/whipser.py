from agent_platform.models.transcript import Transcript

class WhisperCppClient:


    def __init__(self, base_url: str):
        self._url = base_url

    async def transcribe(self, audio: bytes, *, source=None) -> Transcript:
        """r = await self._http.post(f"{self._url}/inference", files={"audio": audio})
        d = r.json()
        return Transcript(content=d["text"], language=Language(d["language"]), audio_segment_id=d["segments"])"""
        ...

    def stream(self, frames): ...