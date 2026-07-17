from agent_platform.core.interfaces.translation.config import TranslationConfig
from agent_platform.integrations.translation.deepl.config import DeepLConfig
from agent_platform.integrations.translation.google_translate.config import (
    GoogleTranslateConfig,
)


class TestTranslationConfig:
    def test_can_be_instantiated(self):
        cfg = TranslationConfig()
        assert isinstance(cfg, TranslationConfig)


class TestDeepLConfig:
    def test_inherits_translation_config(self):
        cfg = DeepLConfig()
        assert isinstance(cfg, TranslationConfig)


class TestGoogleTranslateConfig:
    def test_inherits_translation_config(self):
        cfg = GoogleTranslateConfig()
        assert isinstance(cfg, TranslationConfig)
