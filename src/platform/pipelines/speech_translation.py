from platform.integrations.speech.port import BaseSpeechToText
from platform.integrations.translation.port import BaseTranslator
from models.language import Language
from models.transcript import Transcript

class SpeechTranslationPipeline:
    def __init__(self, stt: BaseSpeechToText, translator: BaseTranslator):
        self._stt = stt
        self._translator = translator

    async def run(self, audio: bytes, target: Language) -> Transcript:
        """transcript = await self._stt.transcribe(audio)
        if transcript.language != target:
            transcript.text = await self._translator.translate(
                transcript.text, target=target, source=transcript.language
            )
        return transcript"""
        ...