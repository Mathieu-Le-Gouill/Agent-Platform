import traceback

import pytest
from pydantic import ValidationError

from agent_platform.config.settings import Settings, get_settings
from agent_platform.core.errors import ConfigError


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

    def test_llm_options_default_to_empty(self):
        assert Settings(_env_file=None).llm_options == {}

    def test_llm_options_parse_from_a_json_object(self, monkeypatch):
        monkeypatch.setenv(
            "AGENT_PLATFORM_LLM_OPTIONS",
            '{"anthropic": {"effort": "high"}, "openai": {"seed": 3}}',
        )

        settings = Settings(_env_file=None)

        assert settings.llm_options == {
            "anthropic": {"effort": "high"},
            "openai": {"seed": 3},
        }

    def test_llm_options_must_map_a_slug_to_an_object(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_LLM_OPTIONS", '{"openai": 3}')

        with pytest.raises(ValidationError):
            Settings(_env_file=None)

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

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("llm_rate_limit", 0),
            ("llm_rate_limit", -1),
            ("llm_rate_limit_burst", 0.5),
            ("mcp_tool_timeout", 0),
        ],
    )
    def test_unusable_rate_limit_or_mcp_timeout_is_rejected(self, field, value):
        with pytest.raises(ValidationError, match=field):
            Settings(_env_file=None, **{field: value})

    def test_mcp_tool_timeout_defaults_to_sixty_seconds(self):
        assert Settings(_env_file=None).mcp_tool_timeout == 60.0

    def test_unknown_env_vars_are_ignored(self, monkeypatch):
        monkeypatch.setenv("AGENT_PLATFORM_SOME_UNKNOWN_FIELD", "value")

        settings = Settings(_env_file=None)

        assert not hasattr(settings, "some_unknown_field")


class TestGetSettings:
    @pytest.fixture
    def isolated_env(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        get_settings.cache_clear()
        yield monkeypatch
        get_settings.cache_clear()

    def test_returns_cached_instance(self):
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()

    def test_a_malformed_setting_names_the_field_and_never_the_value(
        self, isolated_env
    ):
        isolated_env.setenv(
            "AGENT_PLATFORM_LLM_OPTIONS", '{"openai": "sk-test-secret"}'
        )
        isolated_env.setenv("AGENT_PLATFORM_LLM_RATE_LIMIT", "-98765")

        with pytest.raises(ConfigError) as caught:
            get_settings()

        message = str(caught.value)
        printed = "".join(traceback.format_exception(caught.value))
        assert "llm_options.openai" in message
        assert "llm_rate_limit" in message
        assert caught.value.__context__ is None
        for value in ("sk-test-secret", "-98765"):
            assert value not in message
            assert value not in printed

    def test_invalid_json_names_the_field(self, isolated_env):
        isolated_env.setenv("AGENT_PLATFORM_LLM_OPTIONS", "{not json")

        with pytest.raises(ConfigError, match="llm_options") as caught:
            get_settings()

        assert "not json" not in "".join(traceback.format_exception(caught.value))

    def test_a_failure_is_not_cached(self, isolated_env):
        isolated_env.setenv("AGENT_PLATFORM_LLM_RATE_LIMIT", "-1")
        with pytest.raises(ConfigError):
            get_settings()

        isolated_env.delenv("AGENT_PLATFORM_LLM_RATE_LIMIT")

        assert get_settings().llm_rate_limit is None
