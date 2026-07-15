import pytest
from pydantic import SecretStr

pytest.importorskip("langchain_mistralai")
pytest.importorskip("langchain_anthropic")

from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig
from agent_platform.integrations.credentials.mistral import MistralCredentials
from agent_platform.integrations.credentials.anthropic import AnthropicCredentials
from agent_platform.core.schemas.message import Prompt, UserMessage


def test_mistral_to_langchain_no_max_retries():
    from agent_platform.integrations.llm.providers.mistral import _to_langchain_mistral

    cfg = MistralGenerationConfig(max_retries=None)
    creds = MistralCredentials()
    result = _to_langchain_mistral(cfg, creds)
    assert "max_retries" in result


def test_anthropic_to_langchain_no_max_retries():
    from agent_platform.integrations.llm.providers.anthropic import (
        _to_langchain_anthropic,
    )
    from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig

    cfg = AnthropicGenerationConfig(max_retries=None)
    creds = AnthropicCredentials()
    result = _to_langchain_anthropic(cfg, creds)
    assert "max_retries" in result


def test_anthropic_to_langchain_none_config():
    from agent_platform.integrations.llm.providers.anthropic import (
        _to_langchain_anthropic,
    )
    from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig

    creds = AnthropicCredentials()
    cfg = AnthropicGenerationConfig()
    result = _to_langchain_anthropic(cfg, creds)
    assert "max_tokens" in result
