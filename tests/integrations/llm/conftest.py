from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_llm_provider():
    from agent_platform.core.interfaces.llm.config import GenerationConfig
    from agent_platform.integrations.llm.langchain_base import (
        LangChainLLMProvider,
    )

    class _TestLLMProvider(LangChainLLMProvider):
        def _client(self, config):
            raise NotImplementedError

        def _tool_to_schema(self, tool):
            raise NotImplementedError

        def _default_config(self):
            return GenerationConfig()

    provider = _TestLLMProvider()
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock()
    mock_model.astream = MagicMock()
    provider._client = MagicMock(return_value=mock_model)
    return provider, mock_model
