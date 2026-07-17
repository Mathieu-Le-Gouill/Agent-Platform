from unittest.mock import patch

import pytest
from pydantic import SecretStr

pytest.importorskip("langchain_anthropic")
pytest.importorskip("langchain_mistralai")
pytest.importorskip("langchain_ollama")
pytest.importorskip("langchain_openai")

from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.credentials import AnthropicCredentials
from agent_platform.integrations.credentials import MistralCredentials
from agent_platform.integrations.credentials import OllamaCredentials
from agent_platform.core.errors import MissingCredentialError


@patch("agent_platform.integrations.llm.ollama.ollama.ChatOllama")
async def test_ollama_client_creation(mock_chat):
    from agent_platform.integrations.llm.ollama.ollama import OllamaLLM

    provider = OllamaLLM(OllamaCredentials())
    config = OllamaGenerationConfig(model="llama3", temperature=0.5)
    client = provider._client(config)
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


@patch("agent_platform.integrations.llm.ollama.ollama.ChatOllama")
async def test_ollama_client_none_config(mock_chat):
    from agent_platform.integrations.llm.ollama.ollama import OllamaLLM

    provider = OllamaLLM(OllamaCredentials())
    config = OllamaGenerationConfig(model="llama3")
    client = provider._client(config)
    mock_chat.assert_called_once()


@patch("agent_platform.integrations.llm.mistral.mistral.ChatMistralAI")
async def test_mistral_client_creation(mock_chat):
    from agent_platform.integrations.llm.mistral.mistral import MistralLLM

    provider = MistralLLM(MistralCredentials(api_key=SecretStr("key")))
    config = MistralGenerationConfig(model="mistral-large", temperature=0.3)
    client = provider._client(config)
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


@patch("agent_platform.integrations.llm.mistral.mistral.ChatMistralAI")
async def test_mistral_client_none_config(mock_chat):
    from agent_platform.integrations.llm.mistral.mistral import MistralLLM

    provider = MistralLLM(MistralCredentials(api_key=SecretStr("key")))
    config = MistralGenerationConfig(model="mistral-large")
    client = provider._client(config)
    mock_chat.assert_called_once()


@patch("agent_platform.integrations.llm.anthropic.anthropic.ChatAnthropic")
async def test_anthropic_client_creation(mock_chat):
    from agent_platform.integrations.llm.anthropic.anthropic import AnthropicLLM

    provider = AnthropicLLM(AnthropicCredentials(api_key=SecretStr("key")))
    config = AnthropicGenerationConfig(model="claude-3", temperature=0.5)
    client = provider._client(config)
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


@patch("agent_platform.integrations.llm.anthropic.anthropic.ChatAnthropic")
async def test_anthropic_client_none_config(mock_chat):
    from agent_platform.integrations.llm.anthropic.anthropic import AnthropicLLM

    provider = AnthropicLLM(AnthropicCredentials(api_key=SecretStr("key")))
    config = AnthropicGenerationConfig(model="claude-3")
    client = provider._client(config)
    mock_chat.assert_called_once()


@patch("agent_platform.integrations.llm.openai.openai.ChatOpenAI")
async def test_openai_client_creation(mock_chat):
    from agent_platform.integrations.llm.openai.openai import OpenAILLM

    provider = OpenAILLM(OpenAICredentials(api_key=SecretStr("key")))
    config = OpenAIGenerationConfig(model="gpt-4", temperature=0.5)
    client = provider._client(config)
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


@patch("agent_platform.integrations.llm.openai.openai.ChatOpenAI")
async def test_openai_client_none_config(mock_chat):
    from agent_platform.integrations.llm.openai.openai import OpenAILLM

    provider = OpenAILLM(OpenAICredentials(api_key=SecretStr("key")))
    config = OpenAIGenerationConfig(model="gpt-4")
    client = provider._client(config)
    mock_chat.assert_called_once()


@patch("agent_platform.integrations.llm.openai.openai.ChatOpenAI")
async def test_openai_client_no_max_retries(mock_chat):
    from agent_platform.integrations.llm.openai.openai import _to_langchain_openai

    cfg = OpenAIGenerationConfig(max_retries=None)
    creds = OpenAICredentials()
    result = _to_langchain_openai(cfg, creds)
    assert "max_retries" in result


# --- MissingCredentialError tests ---


@patch("agent_platform.integrations.llm.anthropic.anthropic.ChatAnthropic")
async def test_anthropic_missing_credential_error(mock_chat):
    from agent_platform.integrations.llm.anthropic.anthropic import AnthropicLLM

    provider = AnthropicLLM()
    provider._credentials = AnthropicCredentials()
    cfg = AnthropicGenerationConfig(model="claude-3")
    with pytest.raises(MissingCredentialError, match="Anthropic API key is required"):
        provider._client(cfg)


@patch("agent_platform.integrations.llm.mistral.mistral.ChatMistralAI")
async def test_mistral_missing_credential_error(mock_chat):
    from agent_platform.integrations.llm.mistral.mistral import MistralLLM

    provider = MistralLLM()
    provider._credentials = MistralCredentials()
    cfg = MistralGenerationConfig(model="mistral-large")
    with pytest.raises(MissingCredentialError, match="Mistral API key is required"):
        provider._client(cfg)


@patch("agent_platform.integrations.llm.openai.openai.ChatOpenAI")
async def test_openai_missing_credential_error(mock_chat):
    from agent_platform.integrations.llm.openai.openai import OpenAILLM

    provider = OpenAILLM()
    provider._credentials = OpenAICredentials()
    cfg = OpenAIGenerationConfig(model="gpt-4")
    with pytest.raises(MissingCredentialError, match="OPENAI API key is required"):
        provider._client(cfg)


async def test_ollama_no_credential_needed():
    from agent_platform.integrations.llm.ollama.ollama import OllamaLLM

    provider = OllamaLLM(OllamaCredentials())
    cfg = OllamaGenerationConfig(model="llama3")
    client = provider._client(cfg)
    assert client is not None
