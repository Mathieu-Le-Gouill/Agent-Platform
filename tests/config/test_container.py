import pytest

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.config.container import build_agent, build_provider
from agent_platform.config.settings import Settings
from agent_platform.core.errors import ConfigError
from agent_platform.integrations import llm as llm_module
from agent_platform.integrations.llm.openai.openai import OpenAILLM


class TestBuildAgent:
    def test_builds_conversation_agent_with_openai(self):
        settings = Settings(
            _env_file=None, llm_provider="OpenAILLM", agent_name="test-agent"
        )

        agent = build_agent(settings)

        assert isinstance(agent, ConversationAgent)
        assert agent.name == "test-agent"

    def test_unknown_provider_raises_config_error(self):
        settings = Settings(_env_file=None, llm_provider="does-not-exist")

        with pytest.raises(ConfigError):
            build_agent(settings)


class TestBuildProvider:
    def test_instantiates_provider_by_class_name(self):
        provider = build_provider(llm_module, "OpenAILLM")

        assert isinstance(provider, OpenAILLM)

    def test_unknown_provider_raises_config_error(self):
        with pytest.raises(ConfigError):
            build_provider(llm_module, "does-not-exist")
