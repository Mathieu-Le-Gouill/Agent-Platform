from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_platform.agents.context import (
    SummarizingStrategy,
    TokenBudgetStrategy,
    TurnCountStrategy,
)
from agent_platform.core.interfaces.llm.response import FinishReason, LLMResponse
from agent_platform.core.schemas.message import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from agent_platform.core.schemas.token import TokenUsage


class TestTurnCountStrategy:
    def test_invalid_max_turns_raises(self):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            TurnCountStrategy(0)

    @pytest.mark.asyncio
    async def test_trim_no_op_when_under_limit(self):
        strategy = TurnCountStrategy(2)
        history = [UserMessage(content="Hi"), AssistantMessage(content="Hello")]
        assert await strategy.trim(history) == history

    @pytest.mark.asyncio
    async def test_trim_drops_oldest_turn(self):
        strategy = TurnCountStrategy(1)
        history = [
            UserMessage(content="Turn 1"),
            AssistantMessage(content="Response 1"),
            UserMessage(content="Turn 2"),
            AssistantMessage(content="Response 2"),
        ]
        trimmed = await strategy.trim(history)
        assert [m.content for m in trimmed] == ["Turn 2", "Response 2"]

    @pytest.mark.asyncio
    async def test_trim_keeps_exact_limit(self):
        strategy = TurnCountStrategy(2)
        history = [
            UserMessage(content="Turn 1"),
            AssistantMessage(content="Response 1"),
            UserMessage(content="Turn 2"),
            AssistantMessage(content="Response 2"),
        ]
        assert await strategy.trim(history) == history


class TestTokenBudgetStrategy:
    def test_invalid_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            TokenBudgetStrategy(0, count_tokens=len)

    @pytest.mark.asyncio
    async def test_no_user_messages_returns_history_unchanged(self):
        strategy = TokenBudgetStrategy(10, count_tokens=len)
        history = [SystemMessage(content="sys")]
        assert await strategy.trim(history) == history

    @pytest.mark.asyncio
    async def test_trim_no_op_when_under_budget(self):
        strategy = TokenBudgetStrategy(1000, count_tokens=len)
        history = [
            UserMessage(content="Turn 1"),
            AssistantMessage(content="Response 1"),
            UserMessage(content="Turn 2"),
            AssistantMessage(content="Response 2"),
        ]
        assert await strategy.trim(history) == history

    @pytest.mark.asyncio
    async def test_trim_drops_oldest_turns_over_budget(self):
        # "Turn 2" + "Response 2" = 6 + 10 = 16 chars, fits a budget of 20;
        # adding "Turn 1"/"Response 1" (6 + 10 more) would exceed it.
        strategy = TokenBudgetStrategy(20, count_tokens=len)
        history = [
            UserMessage(content="Turn 1"),
            AssistantMessage(content="Response 1"),
            UserMessage(content="Turn 2"),
            AssistantMessage(content="Response 2"),
        ]
        trimmed = await strategy.trim(history)
        assert [m.content for m in trimmed] == ["Turn 2", "Response 2"]

    @pytest.mark.asyncio
    async def test_always_keeps_most_recent_turn_even_if_over_budget(self):
        strategy = TokenBudgetStrategy(1, count_tokens=len)
        history = [
            UserMessage(content="Turn 1"),
            AssistantMessage(content="A much longer response that exceeds budget"),
        ]
        trimmed = await strategy.trim(history)
        assert trimmed == history


class TestSummarizingStrategy:
    def test_invalid_max_turns_raises(self):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            SummarizingStrategy(MagicMock(), max_turns=0)

    @pytest.mark.asyncio
    async def test_no_op_when_under_limit(self):
        strategy = SummarizingStrategy(MagicMock(), max_turns=2)
        history = [UserMessage(content="Hi"), AssistantMessage(content="Hello")]
        assert await strategy.trim(history) == history

    @pytest.mark.asyncio
    async def test_summarizes_dropped_turns_into_system_message(self):
        llm = MagicMock()
        llm.agenerate = AsyncMock(
            return_value=LLMResponse(
                message=AssistantMessage(content="Summary of turn 1"),
                usage=TokenUsage(),
                model="test",
                finish_reason=FinishReason.STOP,
            )
        )
        strategy = SummarizingStrategy(llm, max_turns=1)
        history = [
            UserMessage(content="Turn 1"),
            AssistantMessage(content="Response 1"),
            UserMessage(content="Turn 2"),
            AssistantMessage(content="Response 2"),
        ]
        trimmed = await strategy.trim(history)

        assert isinstance(trimmed[0], SystemMessage)
        assert trimmed[0].content == "Summary of turn 1"
        assert [m.content for m in trimmed[1:]] == ["Turn 2", "Response 2"]
        llm.agenerate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_summarizes_dropped_tool_messages_too(self):
        llm = MagicMock()
        llm.agenerate = AsyncMock(
            return_value=LLMResponse(
                message=AssistantMessage(content="Summary including a tool call"),
                usage=TokenUsage(),
                model="test",
                finish_reason=FinishReason.STOP,
            )
        )
        strategy = SummarizingStrategy(llm, max_turns=1)
        history = [
            UserMessage(content="Turn 1"),
            ToolMessage(
                result=ToolResult(tool_call_id="c1", name="tool", content="tool output")
            ),
            AssistantMessage(content="Response 1"),
            UserMessage(content="Turn 2"),
            AssistantMessage(content="Response 2"),
        ]
        trimmed = await strategy.trim(history)
        assert trimmed[0].content == "Summary including a tool call"
        transcript = llm.agenerate.call_args.kwargs["prompt"].last_user_message().text
        assert "tool output" in transcript

    @pytest.mark.asyncio
    async def test_summary_unavailable_when_llm_returns_no_message(self):
        llm = MagicMock()
        llm.agenerate = AsyncMock(
            return_value=LLMResponse(
                message=None,
                usage=TokenUsage(),
                model="test",
                finish_reason=FinishReason.STOP,
            )
        )
        strategy = SummarizingStrategy(llm, max_turns=1)
        history = [
            UserMessage(content="Turn 1"),
            AssistantMessage(content="Response 1"),
            UserMessage(content="Turn 2"),
            AssistantMessage(content="Response 2"),
        ]
        trimmed = await strategy.trim(history)
        assert trimmed[0].content == "(summary unavailable)"
