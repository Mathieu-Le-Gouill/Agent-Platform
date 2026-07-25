from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from agent_platform.agents.tools import SummarizeInput, SummarizeTool, ToolError
from agent_platform.core.interfaces.llm.response import LLMResponse
from agent_platform.core.schemas.message import AssistantMessage
from agent_platform.core.schemas.token import TokenUsage


def _response(text: str) -> LLMResponse:
    return LLMResponse(
        message=AssistantMessage(content=text),
        usage=TokenUsage(),
        model="test-model",
    )


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.agenerate = AsyncMock(return_value=_response("A short summary."))
    return llm


@pytest.fixture
def tool(mock_llm):
    return SummarizeTool(llm=mock_llm)


class TestSummarizeInput:
    def test_valid_input(self):
        inp = SummarizeInput(text="some long text")
        assert inp.text == "some long text"
        assert inp.max_words is None

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError):
            SummarizeInput(text="")

    def test_max_words_must_be_positive(self):
        with pytest.raises(ValidationError):
            SummarizeInput(text="hi", max_words=0)


class TestSummarizeTool:
    def test_name_and_description(self, tool):
        assert tool.name == "summarize"
        assert tool.description

    async def test_run_success(self, tool, mock_llm):
        result = await tool.run(text="some long text")
        assert result == "A short summary."
        mock_llm.agenerate.assert_awaited_once()
        prompt = mock_llm.agenerate.await_args.args[0]
        assert prompt.last_user_message().text == "some long text"

    async def test_run_with_max_words_in_system_prompt(self, tool, mock_llm):
        await tool.run(text="some long text", max_words=20)
        prompt = mock_llm.agenerate.await_args.args[0]
        assert "20 words" in prompt.system_prompt()

    async def test_run_provider_error_wrapped(self, tool, mock_llm):
        mock_llm.agenerate = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(ToolError, match="Summarization failed"):
            await tool.run(text="some long text")

    async def test_run_empty_response_raises(self, tool, mock_llm):
        mock_llm.agenerate = AsyncMock(return_value=_response(""))
        with pytest.raises(ToolError, match="no content"):
            await tool.run(text="some long text")

    async def test_run_missing_text_raises(self, tool):
        with pytest.raises(ValidationError):
            await tool.run()
