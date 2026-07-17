from unittest.mock import AsyncMock, MagicMock

from agent_platform.integrations.llm.langchain_base import LangChainLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.schemas.message import Prompt, UserMessage


async def test_stream_content_is_none():
    """Cover langchain_base branch: content is neither str nor list."""

    class _Provider(LangChainLLMProvider):
        def _client(self, config):
            raise NotImplementedError

        def _tool_to_schema(self, tool):
            raise NotImplementedError

        def _default_config(self):
            return GenerationConfig()

    chunk = MagicMock(spec=[])
    chunk.content = None
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
