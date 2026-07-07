from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from agent_platform.integrations.llm.config import GenerationConfig
from agent_platform.models.message import Prompt, UserMessage


class _TestConcreteLLM:
    """Helper that patches the langchain base _client to return a mock."""

    def __init__(self):
        from agent_platform.integrations.llm.providers.openai import OpenAILLM

        self.provider = OpenAILLM(api_key=SecretStr("sk-test"))
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
    result = await c.provider.generate(prompt, model="gpt-4")

    assert result.message.content == "Hello world"
    assert result.model == "gpt-4"


@patch("agent_platform.integrations.llm.providers.openai.ChatOpenAI")
async def test_generate_with_openai_client(mock_chat):
    mock_instance = MagicMock()
    mock_chat.return_value = mock_instance
    mock_response = MagicMock()
    mock_response.content = "Response"
    mock_response.tool_calls = None
    mock_response.usage_metadata = None
    mock_instance.ainvoke = AsyncMock(return_value=mock_response)

    from agent_platform.integrations.llm.providers.openai import OpenAILLM

    provider = OpenAILLM(api_key=SecretStr("sk-test"))

    prompt = Prompt(messages=[UserMessage(content="Hi")])
    result = await provider.generate(prompt, model="gpt-4")

    assert result.message.content == "Response"
    mock_chat.assert_called_once()
    mock_instance.ainvoke.assert_called_once()


async def test_stream_handles_non_str_non_dict_items():
    from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider
    from agent_platform.models.message import UserMessage, Prompt

    class _Provider(LangChainLLMProvider):
        def _client(self, model, config=None):
            raise NotImplementedError

        def _tool_to_schema(self, tool):
            raise NotImplementedError

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
    results = [c async for c in provider.stream(prompt, model="test")]

    assert len(results) == 1
    assert results[0].delta == ""
    assert results[0].finish_reason is not None
