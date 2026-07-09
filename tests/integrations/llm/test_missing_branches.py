import pytest
from pydantic import SecretStr

pytest.importorskip("langchain_mistralai")
pytest.importorskip("langchain_anthropic")

from agent_platform.integrations.llm.config import (
    GenerationConfig,
    MistralGenerationConfig,
)
from agent_platform.integrations.credentials import (
    MistralCredentials,
    AnthropicCredentials,
)
from agent_platform.models.message import Prompt, UserMessage


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
    from agent_platform.integrations.llm.config import AnthropicGenerationConfig

    cfg = AnthropicGenerationConfig(max_retries=None)
    creds = AnthropicCredentials()
    result = _to_langchain_anthropic(cfg, creds)
    assert "max_retries" in result


def test_anthropic_to_langchain_none_config():
    from agent_platform.integrations.llm.providers.anthropic import (
        _to_langchain_anthropic,
    )
    from agent_platform.integrations.llm.config import AnthropicGenerationConfig

    creds = AnthropicCredentials()
    cfg = AnthropicGenerationConfig()
    result = _to_langchain_anthropic(cfg, creds)
    assert "max_tokens" in result
