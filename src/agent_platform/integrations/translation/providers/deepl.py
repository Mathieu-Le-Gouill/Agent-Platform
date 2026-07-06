from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import deepl

if TYPE_CHECKING:
    from deepl import TextResult

from agent_platform.integrations.translation.base import BaseTranslator
from agent_platform.models.chunk import TextChunk
from agent_platform.models.enums import Language


_DEEPL_TARGETS: dict[Language, str] = {
    Language.EN: "EN-US",
}


class DeepLTranslator(BaseTranslator):
    def __init__(self, auth_key: str) -> None:
        self._client = deepl.Translator(auth_key)

    async def translate(
        self,
        content: TextChunk,
        target: Language,
        source: Language | None = None,
    ) -> TextChunk:
        target_lang = _DEEPL_TARGETS.get(target, target.value)
        source_lang = _DEEPL_TARGETS.get(source) if source else None

        raw = await asyncio.to_thread(
            self._client.translate_text,
            content.text,
            target_lang=target_lang,
            source_lang=source_lang,
        )

        result: TextResult = raw[0] if isinstance(raw, list) else raw

        return TextChunk(
            text=result.text,
            metadata={
                **content.metadata,
                "translation_provider": "deepl",
                "detected_source_lang": result.detected_source_lang,
            },
        )
