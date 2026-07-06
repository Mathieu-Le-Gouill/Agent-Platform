import pytest
from pydantic import SecretStr

from agent_platform.integrations.llm.config import GenerationConfig
from agent_platform.models.message import Prompt, UserMessage


def test_mistral_to_langchain_no_max_retries():
    from agent_platform.integrations.llm.providers.mistral import _to_langchain_mistral

    cfg = GenerationConfig(max_retries=None)
    result = _to_langchain_mistral(cfg)
    assert "max_retries" not in result


def test_anthropic_to_langchain_no_max_retries():
    from agent_platform.integrations.llm.providers.anthropic import (
        _to_langchain_anthropic,
    )

    cfg = GenerationConfig(max_retries=None)
    result = _to_langchain_anthropic(cfg)
    assert "max_retries" not in result


def test_anthropic_to_langchain_none_config():
    from agent_platform.integrations.llm.providers.anthropic import (
        _to_langchain_anthropic,
    )

    result = _to_langchain_anthropic(None)
    assert result == {"max_tokens": 1024}
