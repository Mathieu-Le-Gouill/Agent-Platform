import pytest
from pydantic import ValidationError

from agent_platform.config.settings import Settings, get_settings


class TestSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("AGENT_PLATFORM_DEFAULT_LLM_MODEL", raising=False)

        settings = Settings(_env_file=None)

        assert settings.default_llm_model == "openai:gpt-4o-mini"
        assert settings.default_image_model == "dalle:dall-e-3"
        assert settings.default_audio_model == "openai:whisper-1"
        assert settings.agent_name == "assistant"
        assert settings.agent_system_prompt is None
        assert settings.max_iterations == 10
        assert settings.log_level == "INFO"
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 8000

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(
            "AGENT_PLATFORM_DEFAULT_LLM_MODEL", "anthropic:claude-sonnet-4-5"
        )
        monkeypatch.setenv("AGENT_PLATFORM_DEFAULT_IMAGE_MODEL", "midjourney:v6")
        monkeypatch.setenv("AGENT_PLATFORM_DEFAULT_AUDIO_MODEL", "deepgram:nova-2")
        monkeypatch.setenv("AGENT_PLATFORM_API_PORT", "9001")

        settings = Settings(_env_file=None)

        assert settings.default_llm_model == "anthropic:claude-sonnet-4-5"
        assert settings.default_image_model == "midjourney:v6"
        assert settings.default_audio_model == "deepgram:nova-2"
        assert settings.api_port == 9001

    def test_api_defaults(self):
        settings = Settings(_env_file=None)

        assert settings.api_keys == []
        assert settings.api_auth_disabled is False
        assert settings.api_max_conversations_per_key == 100

    def test_api_keys_parse_from_a_json_list(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_API_KEYS", '["key-1", "key-2"]')
        monkeypatch.setenv("AGENT_PLATFORM_API_MAX_CONVERSATIONS_PER_KEY", "5")

        settings = Settings(_env_file=None)

        assert [k.get_secret_value() for k in settings.api_keys] == ["key-1", "key-2"]
        assert settings.api_max_conversations_per_key == 5

    def test_api_keys_are_not_shown_in_repr(self):
        settings = Settings(_env_file=None, api_keys=["key-1"])

        assert "key-1" not in repr(settings)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_api_key_is_rejected(self, blank):
        with pytest.raises(ValidationError, match="api_keys"):
            Settings(_env_file=None, api_keys=["good", blank])

    def test_conversation_cap_below_one_is_rejected(self):
        with pytest.raises(ValidationError, match="api_max_conversations_per_key"):
            Settings(_env_file=None, api_max_conversations_per_key=0)

    def test_unknown_env_vars_are_ignored(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_SOME_UNKNOWN_FIELD", "value")

        settings = Settings(_env_file=None)

        assert not hasattr(settings, "some_unknown_field")


class TestGetSettings:
    def test_returns_cached_instance(self):
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()
