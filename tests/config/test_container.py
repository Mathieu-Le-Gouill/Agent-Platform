import pytest

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.config.container import (
    build_agent,
    build_provider,
    build_provider_from_model_string,
)
from agent_platform.config.settings import Settings
from agent_platform.core.errors import ConfigError
from agent_platform.integrations import llm as llm_module
from agent_platform.integrations.llm.openai.provider import OpenAILLM


class TestBuildAgent:
    def test_builds_conversation_agent_with_openai(self):
        settings = Settings(
            _env_file=None,
            default_llm_model="openai:gpt-4o-mini",
            agent_name="test-agent",
        )

        agent = build_agent(settings)

        assert isinstance(agent, ConversationAgent)
        assert agent.name == "test-agent"

    def test_registers_image_and_audio_tools(self):
        settings = Settings(_env_file=None)

        agent = build_agent(settings)

        assert set(agent.tool_registry.all()) == {"generate_image", "transcribe"}

    def test_unknown_provider_raises_config_error(self):
        settings = Settings(_env_file=None, default_llm_model="does-not-exist:gpt-4o")

        with pytest.raises(ConfigError):
            build_agent(settings)

    def test_malformed_model_string_raises_config_error(self):
        settings = Settings(_env_file=None, default_llm_model="gpt-4o-mini")

        with pytest.raises(ConfigError):
            build_agent(settings)


class TestBuildProvider:
    def test_instantiates_provider_by_alias(self):
        provider = build_provider(llm_module, "openai")

        assert isinstance(provider, OpenAILLM)

    def test_instantiates_provider_by_class_name(self):
        provider = build_provider(llm_module, "OpenAILLM")

        assert isinstance(provider, OpenAILLM)

    def test_unknown_provider_raises_config_error(self):
        with pytest.raises(ConfigError):
            build_provider(llm_module, "does-not-exist")


class TestBuildProviderFromModelString:
    def test_resolves_provider_and_model(self):
        provider, model = build_provider_from_model_string(
            llm_module, "openai:gpt-4o-mini"
        )

        assert isinstance(provider, OpenAILLM)
        assert model == "gpt-4o-mini"

    def test_malformed_string_raises_config_error(self):
        with pytest.raises(ConfigError):
            build_provider_from_model_string(llm_module, "gpt-4o-mini")

    def test_unknown_provider_raises_config_error(self):
        with pytest.raises(ConfigError):
            build_provider_from_model_string(llm_module, "does-not-exist:gpt-4o-mini")
