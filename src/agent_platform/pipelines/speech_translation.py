from agent_platform.core.interfaces.speech.base import BaseSpeechToText
from agent_platform.core.interfaces.translation.base import BaseTranslator
from agent_platform.core.schemas.conversation import Transcript
from agent_platform.core.schemas.enums import Language


class SpeechTranslationPipeline:
    def __init__(self, stt: BaseSpeechToText, translator: BaseTranslator) -> None:
        self._stt = stt
        self._translator = translator

    async def run(self, audio: bytes, target: Language) -> Transcript:
        """transcript = await self._stt.transcribe(audio)
        if transcript.language != target:
            transcript.text = await self._translator.translate(
                transcript.text, target=target, source=transcript.language
            )
        return transcript"""
        raise NotImplementedError
