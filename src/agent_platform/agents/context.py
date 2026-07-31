from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.schemas.message import (
    ContentMessage,
    Message,
    Prompt,
    SystemMessage,
    UserMessage,
)


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


def _message_text(message: Message) -> str:
    if isinstance(message, ContentMessage):
        return message.text
    return message.result.content


def _turn_bounds(history: list[Message]) -> tuple[list[int], list[int]]:
    user_indices = [i for i, m in enumerate(history) if isinstance(m, UserMessage)]
    return user_indices, [*user_indices, len(history)]


class TokenBudgetStrategy:
    """Drops the oldest whole turns until the remaining history fits `max_tokens`.

    Always keeps at least the most recent turn, even if it alone exceeds the
    budget, since dropping everything would leave the model with no context at
    all. `count_tokens` is injected since tokenization is model-specific.
    """

    def __init__(self, max_tokens: int, count_tokens: Callable[[str], int]) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._max_tokens = max_tokens
        self._count_tokens = count_tokens

    def trim(self, history: list[Message]) -> list[Message]:
        user_indices, bounds = _turn_bounds(history)
        if not user_indices:
            return history

        kept_from = len(history)
        total = 0
        for i in range(len(user_indices) - 1, -1, -1):
            start, end = bounds[i], bounds[i + 1]
            turn_tokens = sum(
                self._count_tokens(_message_text(m)) for m in history[start:end]
            )
            if kept_from != len(history) and total + turn_tokens > self._max_tokens:
                break
            total += turn_tokens
            kept_from = start

        return history[kept_from:]


class SummarizingStrategy:
    """Like `TurnCountStrategy`, but collapses dropped turns into a summary
    instead of discarding them outright, via a synchronous `BaseLLMProvider`
    call (`generate()`, not `agenerate()`) so `trim()` can stay synchronous
    and satisfy `ContextStrategy` without changing that protocol or
    `ConversationAgent`.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        *,
        max_turns: int,
        summary_prompt: str = (
            "Summarize the following conversation turns concisely, "
            "preserving any facts that matter for later turns:"
        ),
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._llm = llm
        self._max_turns = max_turns
        self._summary_prompt = summary_prompt

    def trim(self, history: list[Message]) -> list[Message]:
        user_indices = [i for i, m in enumerate(history) if isinstance(m, UserMessage)]
        if len(user_indices) <= self._max_turns:
            return history

        first_to_keep = user_indices[-self._max_turns]
        dropped = history[:first_to_keep]
        kept = history[first_to_keep:]

        return [SystemMessage(content=self._summarize(dropped)), *kept]

    def _summarize(self, dropped: list[Message]) -> str:
        transcript = "\n".join(f"{m.role.value}: {_message_text(m)}" for m in dropped)
        response = self._llm.generate(
            prompt=Prompt.build(system=self._summary_prompt, user=transcript)
        )
        if response.message is None:
            return "(summary unavailable)"
        return response.message.text
