import pytest

import agent_platform.integrations.moderation as moderation_module


class TestModerationInitExports:
    def test_openai_moderation_is_lazily_importable(self):
        cls = moderation_module.OpenAIModeration
        assert cls.__name__ == "OpenAIModeration"

    def test_local_moderation_is_lazily_importable(self):
        cls = moderation_module.LocalModeration
        assert cls.__name__ == "LocalModeration"

    def test_unknown_attribute_raises(self):
        with pytest.raises(AttributeError):
            moderation_module.NotAProvider

    def test_dir_includes_providers(self):
        assert "OpenAIModeration" in dir(moderation_module)
        assert "LocalModeration" in dir(moderation_module)
