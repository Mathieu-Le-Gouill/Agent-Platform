from agent_platform.integrations.speech.base import BaseSpeechToText
from agent_platform.integrations.translation.base import BaseTranslator
from agent_platform.models.enums import Language
from agent_platform.models.conversation import Transcript

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