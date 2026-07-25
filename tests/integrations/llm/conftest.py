import pytest

from tests.helpers import make_provider_with_mock_client


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

    return make_provider_with_mock_client(
        _TestLLMProvider, async_methods=("ainvoke",), sync_methods=("astream",)
    )
