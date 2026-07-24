from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

pytest.importorskip("langchain_openai")

from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.schemas.message import Prompt, UserMessage
from agent_platform.integrations.credentials import OpenAICredentials
from agent_platform.integrations.llm.openai.config import OpenAIGenerationConfig


class _TestConcreteLLM:
    """Helper that patches the langchain base _client to return a mock."""

    def __init__(self):
        from agent_platform.integrations.llm.openai.openai import OpenAILLM

        self.provider = OpenAILLM(OpenAICredentials(api_key=SecretStr("sk-test")))
        self.mock_client = MagicMock()
        self.provider._client = MagicMock(return_value=self.mock_client)


async def test_generate():
    c = _TestConcreteLLM()
    mock_response = MagicMock()
    mock_response.content = "Hello world"
    mock_response.tool_calls = None
    mock_response.usage_metadata = None
    c.mock_client.ainvoke = AsyncMock(return_value=mock_response)

    prompt = Prompt(messages=[UserMessage(content="Hi")])
    config = OpenAIGenerationConfig(model="gpt-4")
    result = await c.provider.agenerate(prompt, config=config)

    assert result.message.content == "Hello world"
    assert result.model == "gpt-4"


async def test_generate_with_openai_client(mocker):
    mock_chat = mocker.patch("agent_platform.integrations.llm.openai.openai.ChatOpenAI")
    mock_instance = MagicMock()
    mock_chat.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.content = "Response"
    mock_response.tool_calls = None
    mock_response.usage_metadata = None
    mock_instance.ainvoke = AsyncMock(return_value=mock_response)

    from agent_platform.integrations.llm.openai.openai import OpenAILLM

    provider = OpenAILLM(OpenAICredentials(api_key=SecretStr("sk-test")))

    prompt = Prompt(messages=[UserMessage(content="Hi")])
    config = OpenAIGenerationConfig(model="gpt-4")
    result = await provider.agenerate(prompt, config=config)

    assert result.message.content == "Response"
    mock_chat.assert_called_once()
    mock_instance.ainvoke.assert_called_once()


async def test_sync_generate():
    from unittest.mock import MagicMock

    from agent_platform.integrations.llm.openai.openai import OpenAILLM

    provider = OpenAILLM(OpenAICredentials(api_key=SecretStr("sk-test")))
    mock_response = MagicMock()
    mock_response.content = "Sync hello"
    mock_response.tool_calls = None
    mock_response.usage_metadata = None
    mock_client = MagicMock()
    mock_client.invoke.return_value = mock_response
    provider._client = MagicMock(return_value=mock_client)

    prompt = Prompt(messages=[UserMessage(content="Hi")])
    config = OpenAIGenerationConfig(model="gpt-4")
    result = provider.generate(prompt, config=config)

    assert result.message.content == "Sync hello"
    assert result.model == "gpt-4"
    mock_client.invoke.assert_called_once()


async def test_stream_handles_non_str_non_dict_items():
    from agent_platform.core.schemas.message import Prompt, UserMessage
    from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider

    class _Provider(LangChainLLMProvider):
        def _client(self, config):
            raise NotImplementedError

        def _tool_to_schema(self, tool):
            raise NotImplementedError

        def _default_config(self):
            from agent_platform.core.interfaces.llm.config import GenerationConfig

            return GenerationConfig()

    chunk = MagicMock(spec=[])
    chunk.content = [1, 2, 3]
    chunk.usage_metadata = None

    async def _gen():
        yield chunk

    mock_model = MagicMock()
    mock_model.astream = MagicMock(return_value=_gen())

    provider = _Provider()
    provider._client = MagicMock(return_value=mock_model)

    prompt = Prompt(messages=[UserMessage(content="Hi")])
    config = GenerationConfig(model="test")
    results = [c async for c in provider.stream(prompt, config=config)]

    assert len(results) == 1
    assert results[0].delta == ""
    assert results[0].finish_reason is not None
