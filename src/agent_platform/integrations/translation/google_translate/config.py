from __future__ import annotations
from typing import Literal

from agent_platform.core.interfaces.translation.config import TranslationConfig


class GoogleTranslateConfig(TranslationConfig):
    # https://cloud.google.com/translate/docs/reference/rest/v2/translate — client
    # defaults to "html" when omitted, so this is always passed explicitly.
    format: Literal["text", "html"] = "text"
    # https://cloud.google.com/translate/docs/reference/rest/v2/translate
    model: str | None = None
