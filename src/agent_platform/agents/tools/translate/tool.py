from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agent_platform.agents.tools.base import Tool
from agent_platform.agents.tools.safe_execution import safe_call
from agent_platform.core.interfaces.translation.base import BaseTranslator
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.enums import Language


class TranslateInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text to translate")
    target: Language = Field(..., description="Target language")
    source: Language | None = Field(
        default=None, description="Source language, auto-detected if omitted"
    )


class TranslateTool(Tool):
    name = "translate"
    description = "Translate text into a target language."
    input_schema = TranslateInput
    output_schema = TextChunk

    def __init__(self, translator: BaseTranslator) -> None:
        self._translator = translator

    async def run(self, **kwargs: Any) -> TextChunk:
        validated = TranslateInput(**kwargs)
        return await safe_call(
            self._translator.atranslate(
                TextChunk(text=validated.text),
                target=validated.target,
                source=validated.source,
            ),
            "Translation failed",
        )
