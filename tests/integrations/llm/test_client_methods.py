import pytest
from pydantic import SecretStr

pytest.importorskip("langchain_anthropic")
pytest.importorskip("langchain_mistralai")
pytest.importorskip("langchain_ollama")
pytest.importorskip("langchain_openai")

from agent_platform.core.errors import MissingCredentialError
from agent_platform.integrations.credentials import (
    AnthropicCredentials,
    MistralCredentials,
    OllamaCredentials,
    OpenAICredentials,
)
from agent_platform.integrations.llm.anthropic.config import AnthropicGenerationConfig
from agent_platform.integrations.llm.mistral.config import MistralGenerationConfig
from agent_platform.integrations.llm.ollama.config import OllamaGenerationConfig
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig


async def test_ollama_client_creation(mocker):
    mock_chat = mocker.patch(
        "agent_platform.integrations.llm.ollama.provider.ChatOllama"
    )
    from agent_platform.integrations.llm.ollama.provider import OllamaLLM

    provider = OllamaLLM(OllamaCredentials())
    config = OllamaGenerationConfig(model="llama3", temperature=0.5)
    client = provider._client(config)
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


async def test_ollama_client_none_config(mocker):
    mock_chat = mocker.patch(
        "agent_platform.integrations.llm.ollama.provider.ChatOllama"
    )
    from agent_platform.integrations.llm.ollama.provider import OllamaLLM

    provider = OllamaLLM(OllamaCredentials())
    config = OllamaGenerationConfig(model="llama3")
    provider._client(config)
    mock_chat.assert_called_once()


async def test_mistral_client_creation(mocker):
    mock_chat = mocker.patch(
        "agent_platform.integrations.llm.mistral.provider.ChatMistralAI"
    )
    from agent_platform.integrations.llm.mistral.provider import MistralLLM

    provider = MistralLLM(MistralCredentials(api_key=SecretStr("key")))
    config = MistralGenerationConfig(model="mistral-large", temperature=0.3)
    client = provider._client(config)
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


async def test_mistral_client_none_config(mocker):
    mock_chat = mocker.patch(
        "agent_platform.integrations.llm.mistral.provider.ChatMistralAI"
    )
    from agent_platform.integrations.llm.mistral.provider import MistralLLM

    provider = MistralLLM(MistralCredentials(api_key=SecretStr("key")))
    config = MistralGenerationConfig(model="mistral-large")
    provider._client(config)
    mock_chat.assert_called_once()


async def test_anthropic_client_creation(mocker):
    mock_chat = mocker.patch(
        "agent_platform.integrations.llm.anthropic.provider.ChatAnthropic"
    )
    from agent_platform.integrations.llm.anthropic.provider import AnthropicLLM

    provider = AnthropicLLM(AnthropicCredentials(api_key=SecretStr("key")))
    config = AnthropicGenerationConfig(model="claude-3", temperature=0.5)
    client = provider._client(config)
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


async def test_anthropic_client_none_config(mocker):
    mock_chat = mocker.patch(
        "agent_platform.integrations.llm.anthropic.provider.ChatAnthropic"
    )
    from agent_platform.integrations.llm.anthropic.provider import AnthropicLLM

    provider = AnthropicLLM(AnthropicCredentials(api_key=SecretStr("key")))
    config = AnthropicGenerationConfig(model="claude-3")
    provider._client(config)
    mock_chat.assert_called_once()


async def test_openai_client_creation(mocker):
    mock_chat = mocker.patch(
        "agent_platform.integrations.llm.openai.provider.ChatOpenAI"
    )
    from agent_platform.integrations.llm.openai.provider import OpenAILLM

    provider = OpenAILLM(OpenAICredentials(api_key=SecretStr("key")))
    config = OpenAIGenerationConfig(model="gpt-4", temperature=0.5)
    client = provider._client(config)
    mock_chat.assert_called_once()
    assert client is mock_chat.return_value


async def test_openai_client_none_config(mocker):
    mock_chat = mocker.patch(
        "agent_platform.integrations.llm.openai.provider.ChatOpenAI"
    )
    from agent_platform.integrations.llm.openai.provider import OpenAILLM

    provider = OpenAILLM(OpenAICredentials(api_key=SecretStr("key")))
    config = OpenAIGenerationConfig(model="gpt-4")
    provider._client(config)
    mock_chat.assert_called_once()


async def test_openai_client_no_max_retries(mocker):
    mocker.patch("agent_platform.integrations.llm.openai.provider.ChatOpenAI")
    from agent_platform.integrations.llm.openai.provider import _to_langchain_openai

    cfg = OpenAIGenerationConfig(max_retries=None)
    creds = OpenAICredentials()
    result = _to_langchain_openai(cfg, creds)
    assert "max_retries" in result


# --- MissingCredentialError tests ---


async def test_anthropic_missing_credential_error(mocker):
    mocker.patch("agent_platform.integrations.llm.anthropic.provider.ChatAnthropic")
    from agent_platform.integrations.llm.anthropic.provider import AnthropicLLM

    provider = AnthropicLLM()
    provider._credentials = AnthropicCredentials()
    cfg = AnthropicGenerationConfig(model="claude-3")
    with pytest.raises(MissingCredentialError, match="Anthropic API key is required"):
        provider._client(cfg)


async def test_mistral_missing_credential_error(mocker):
    mocker.patch("agent_platform.integrations.llm.mistral.provider.ChatMistralAI")
    from agent_platform.integrations.llm.mistral.provider import MistralLLM

    provider = MistralLLM()
    provider._credentials = MistralCredentials()
    cfg = MistralGenerationConfig(model="mistral-large")
    with pytest.raises(MissingCredentialError, match="Mistral API key is required"):
        provider._client(cfg)


async def test_openai_missing_credential_error(mocker):
    mocker.patch("agent_platform.integrations.llm.openai.provider.ChatOpenAI")
    from agent_platform.integrations.llm.openai.provider import OpenAILLM

    provider = OpenAILLM()
    provider._credentials = OpenAICredentials()
    cfg = OpenAIGenerationConfig(model="gpt-4")
    with pytest.raises(MissingCredentialError, match="OPENAI API key is required"):
        provider._client(cfg)


async def test_ollama_no_credential_needed():
    from agent_platform.integrations.llm.ollama.provider import OllamaLLM

    provider = OllamaLLM(OllamaCredentials())
    cfg = OllamaGenerationConfig(model="llama3")
    client = provider._client(cfg)
    assert client is not None
