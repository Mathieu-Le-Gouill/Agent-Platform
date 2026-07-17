from __future__ import annotations
from agent_platform.core.interfaces.translation.config import TranslationConfig


class DeepLConfig(TranslationConfig):
    formality: str | None = None
