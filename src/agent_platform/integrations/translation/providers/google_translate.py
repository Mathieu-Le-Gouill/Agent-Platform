from __future__ import annotations

from googletrans import Translator as GoogleTranslatorClient

from agent_platform.integrations.translation.base import BaseTranslator
from agent_platform.models.chunk import TextChunk
from agent_platform.models.enums import Language


_GOOGLE_TARGETS: dict[Language, str] = {
    Language.CH: "zh-cn",
}


class GoogleTranslator(BaseTranslator):
    def __init__(self) -> None:
        self._client = GoogleTranslatorClient()

    async def translate(
        self,
        content: TextChunk,
        target: Language,
        source: Language | None = None,
    ) -> TextChunk:
        target_lang = _GOOGLE_TARGETS.get(target, target.value)
        source_lang = _GOOGLE_TARGETS.get(source) if source else None

        result = await self._client.translate(
            content.text,
            dest=target_lang,
            src=source_lang or "",
        )

        return TextChunk(
            text=result.text,
            metadata={
                **content.metadata,
                "translation_provider": "google",
                "detected_source_lang": result.src,
            },
        )
