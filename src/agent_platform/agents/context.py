from __future__ import annotations

from typing import Protocol

from agent_platform.core.schemas.message import Message, UserMessage


class ContextStrategy(Protocol):
    def trim(self, history: list[Message]) -> list[Message]: ...


class TurnCountStrategy:
    def __init__(self, max_turns: int) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._max_turns = max_turns

    def trim(self, history: list[Message]) -> list[Message]:
        user_indices = [i for i, m in enumerate(history) if isinstance(m, UserMessage)]
        if len(user_indices) <= self._max_turns:
            return history
        first_to_keep = user_indices[-self._max_turns]
        return history[first_to_keep:]
