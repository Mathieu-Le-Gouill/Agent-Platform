from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from agent_platform.integrations.llm.config import GenerationConfig, OpenAIConfig
from agent_platform.models.message import Prompt, UserMessage


@patch("agent_platform.integrations.llm.providers.ollama.ChatOllama")
async def test_ollama_client_creation(mock_chat):
    from agent_platform.integrations.llm.providers.ollama import OllamaLLM

    provider = OllamaLLM()
    client = provider._client("llama3", GenerationConfig(temperature=0.5))
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


@patch("agent_platform.integrations.llm.providers.ollama.ChatOllama")
async def test_ollama_client_none_config(mock_chat):
    from agent_platform.integrations.llm.providers.ollama import OllamaLLM

    provider = OllamaLLM()
    client = provider._client("llama3", None)
    mock_chat.assert_called_once()


@patch("agent_platform.integrations.llm.providers.mistral.ChatMistralAI")
async def test_mistral_client_creation(mock_chat):
    from agent_platform.integrations.llm.providers.mistral import MistralLLM

    provider = MistralLLM(api_key=SecretStr("key"))
    client = provider._client("mistral-large", GenerationConfig(temperature=0.3))
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


@patch("agent_platform.integrations.llm.providers.mistral.ChatMistralAI")
async def test_mistral_client_none_config(mock_chat):
    from agent_platform.integrations.llm.providers.mistral import MistralLLM

    provider = MistralLLM(api_key=SecretStr("key"))
    client = provider._client("mistral-large", None)
    mock_chat.assert_called_once()


@patch("agent_platform.integrations.llm.providers.anthropic.ChatAnthropic")
async def test_anthropic_client_creation(mock_chat):
    from agent_platform.integrations.llm.providers.anthropic import AnthropicLLM

    provider = AnthropicLLM(api_key=SecretStr("key"))
    client = provider._client("claude-3", GenerationConfig(temperature=0.5))
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


@patch("agent_platform.integrations.llm.providers.anthropic.ChatAnthropic")
async def test_anthropic_client_none_config(mock_chat):
    from agent_platform.integrations.llm.providers.anthropic import AnthropicLLM

    provider = AnthropicLLM(api_key=SecretStr("key"))
    client = provider._client("claude-3", None)
    mock_chat.assert_called_once()


@patch("agent_platform.integrations.llm.providers.openai.ChatOpenAI")
async def test_openai_client_creation(mock_chat):
    from agent_platform.integrations.llm.providers.openai import OpenAILLM

    provider = OpenAILLM(api_key=SecretStr("key"))
    client = provider._client("gpt-4", GenerationConfig(temperature=0.5))
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


@patch("agent_platform.integrations.llm.providers.openai.ChatOpenAI")
async def test_openai_client_none_config(mock_chat):
    from agent_platform.integrations.llm.providers.openai import OpenAILLM

    provider = OpenAILLM(api_key=SecretStr("key"))
    client = provider._client("gpt-4", None)
    mock_chat.assert_called_once()


@patch("agent_platform.integrations.llm.providers.openai.ChatOpenAI")
async def test_openai_client_no_max_retries(mock_chat):
    from agent_platform.integrations.llm.providers.openai import _to_langchain_openai

    cfg = OpenAIConfig(max_retries=None)
    result = _to_langchain_openai(cfg)
    assert "max_retries" not in result
