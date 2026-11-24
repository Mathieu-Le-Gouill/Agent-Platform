from __future__ import annotations

import asyncio
import logging
from uuid import UUID, uuid4

from agent_platform.agents.agent import Agent
from agent_platform.agents.context import ContextStrategy, TurnCountStrategy
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.cost import CostEstimator
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.persistence import Checkpointer
from agent_platform.core.schemas.message import (
    AssistantMessage,
    Message,
    UserMessage,
)
from agent_platform.core.token_usage import TokenUsageAggregator

_logger = logging.getLogger(__name__)


def _without_dangling_tool_call(messages: list[Message]) -> list[Message]:
    """Return `messages` without a trailing assistant message that asks for
    tool calls: being last, its calls have no results."""
    if messages and isinstance(messages[-1], AssistantMessage):
        if messages[-1].tool_calls:
            return messages[:-1]
    return messages


class ConversationAgent(Agent):
    def __init__(
        self,
        *,
        name: str,
        llm: BaseLLMProvider,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        model: str = "",
        generation_config: GenerationConfig | None = None,
        max_iterations: int = 10,
        max_history_turns: int | None = None,
        context_strategy: ContextStrategy | None = None,
        checkpointer: Checkpointer[list[Message]] | None = None,
        conversation_id: UUID | None = None,
        token_usage_aggregator: TokenUsageAggregator | None = None,
        cost_estimator: CostEstimator | None = None,
    ) -> None:
        if context_strategy is not None and max_history_turns is not None:
            raise ValueError(
                "pass either max_history_turns or context_strategy, not both"
            )
        conversation_id = conversation_id or uuid4()
        super().__init__(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            model=model,
            generation_config=generation_config,
            token_usage_aggregator=token_usage_aggregator,
            usage_key=str(conversation_id),
            cost_estimator=cost_estimator,
        )
        self._history: list[Message] = []
        self._executor = AgentExecutor(self, max_iterations=max_iterations)
        self._context_strategy = context_strategy or (
            TurnCountStrategy(max_history_turns)
            if max_history_turns is not None
            else None
        )
        self._checkpointer = checkpointer
        self._conversation_id = conversation_id

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    @property
    def conversation_id(self) -> UUID:
        return self._conversation_id

    def clear_history(self) -> None:
        self._history.clear()

    def add_user_message(self, content: str) -> None:
        self._history.append(UserMessage(content=content))

    async def chat(self, user_input: str) -> str:
        """Answer `user_input` and keep the turn in the history.

        When the turn fails, the tool calls that already ran stay in the
        history, with their results, so sending the message again does not
        run them again. A tool call whose results never arrived is dropped,
        since a provider rejects a call without a result. The original error
        still propagates.
        """
        self.add_user_message(user_input)

        working = list(self._history)
        try:
            result = await self._executor.run_in_place(
                working, conversation_id=str(self._conversation_id)
            )
        except asyncio.CancelledError:
            self._history = _without_dangling_tool_call(working)
            raise
        except Exception:
            self._history = _without_dangling_tool_call(working)
            await self._persist_after_failure()
            raise

        self._history = working
        await self._truncate_history()

        if self._checkpointer is not None:
            await self._checkpointer.save(
                str(self._conversation_id), list(self._history)
            )

        return result

    async def _persist_after_failure(self) -> None:
        """Trim and save the history of a failed turn; a failure here is logged
        so it never replaces the error the turn failed with."""
        try:
            await self._truncate_history()
            if self._checkpointer is not None:
                await self._checkpointer.save(
                    str(self._conversation_id), list(self._history)
                )
        except Exception:
            _logger.exception(
                "Could not save the history of failed conversation '%s'",
                self._conversation_id,
            )

    async def _truncate_history(self) -> None:
        if self._context_strategy is None:
            return
        self._history = await self._context_strategy.trim(self._history)

    def with_conversation(self, conversation_id: UUID) -> ConversationAgent:
        """Returns a new agent with this agent's configuration and an empty history.

        The fork shares the LLM, tools and checkpointer but gets its own
        `TokenUsageAggregator` (only when this agent has one), so its
        `token_usage` is that conversation's total and dropping the fork frees it.
        """
        return type(self)(
            name=self.name,
            llm=self._llm,
            tool_registry=self._tool_registry,
            system_prompt=self._system_prompt,
            model=self._model,
            generation_config=self._generation_config,
            max_iterations=self._executor.max_iterations,
            context_strategy=self._context_strategy,
            checkpointer=self._checkpointer,
            conversation_id=conversation_id,
            token_usage_aggregator=(
                TokenUsageAggregator() if self._usage_aggregator is not None else None
            ),
            cost_estimator=self._cost_estimator,
        )

    @classmethod
    async def resume(
        cls,
        *,
        conversation_id: UUID,
        checkpointer: Checkpointer[list[Message]],
        name: str,
        llm: BaseLLMProvider,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        model: str = "",
        generation_config: GenerationConfig | None = None,
        max_iterations: int = 10,
        max_history_turns: int | None = None,
        context_strategy: ContextStrategy | None = None,
        token_usage_aggregator: TokenUsageAggregator | None = None,
        cost_estimator: CostEstimator | None = None,
    ) -> ConversationAgent:
        agent = cls(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            model=model,
            generation_config=generation_config,
            max_iterations=max_iterations,
            max_history_turns=max_history_turns,
            context_strategy=context_strategy,
            checkpointer=checkpointer,
            conversation_id=conversation_id,
            token_usage_aggregator=token_usage_aggregator,
            cost_estimator=cost_estimator,
        )
        saved = await checkpointer.load(str(conversation_id))
        if saved is not None:
            agent._history = list(saved)
        return agent
