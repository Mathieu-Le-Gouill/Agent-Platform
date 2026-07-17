from __future__ import annotations
from agent_platform.core.interfaces.translation.config import TranslationConfig


class AzureTranslatorConfig(TranslationConfig):
    api_version: str = "3.0"
