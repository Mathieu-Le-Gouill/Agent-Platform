import pytest

from agent_platform.agents.context import TurnCountStrategy
from agent_platform.core.schemas.message import AssistantMessage, UserMessage


class TestTurnCountStrategy:
    def test_invalid_max_turns_raises(self):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            TurnCountStrategy(0)

    def test_trim_no_op_when_under_limit(self):
        strategy = TurnCountStrategy(2)
        history = [UserMessage(content="Hi"), AssistantMessage(content="Hello")]
        assert strategy.trim(history) == history

    def test_trim_drops_oldest_turn(self):
        strategy = TurnCountStrategy(1)
        history = [
            UserMessage(content="Turn 1"),
            AssistantMessage(content="Response 1"),
            UserMessage(content="Turn 2"),
            AssistantMessage(content="Response 2"),
        ]
        trimmed = strategy.trim(history)
        assert [m.content for m in trimmed] == ["Turn 2", "Response 2"]

    def test_trim_keeps_exact_limit(self):
        strategy = TurnCountStrategy(2)
        history = [
            UserMessage(content="Turn 1"),
            AssistantMessage(content="Response 1"),
            UserMessage(content="Turn 2"),
            AssistantMessage(content="Response 2"),
        ]
        assert strategy.trim(history) == history
