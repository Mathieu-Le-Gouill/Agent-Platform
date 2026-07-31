from __future__ import annotations

from uuid import UUID, uuid4

from agent_platform.agents.agent import Agent
from agent_platform.agents.context import ContextStrategy, TurnCountStrategy
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
from agent_platform.core.persistence import Checkpointer
from agent_platform.core.schemas.message import (
    Message,
    UserMessage,
)


class ConversationAgent(Agent):
    def __init__(
        self,
        *,
        name: str,
        llm: BaseLLMProvider,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        model: str = "default",
        generation_config: GenerationConfig | None = None,
        max_iterations: int = 10,
        max_history_turns: int | None = None,
        context_strategy: ContextStrategy | None = None,
        checkpointer: Checkpointer[list[Message]] | None = None,
        conversation_id: UUID | None = None,
    ) -> None:
        if context_strategy is not None and max_history_turns is not None:
            raise ValueError(
                "pass either max_history_turns or context_strategy, not both"
            )
        super().__init__(
            name=name,
            llm=llm,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
            model=model,
            generation_config=generation_config,
        )
        self._history: list[Message] = []
        self._executor = AgentExecutor(self, max_iterations=max_iterations)
        self._context_strategy = context_strategy or (
            TurnCountStrategy(max_history_turns)
            if max_history_turns is not None
            else None
        )
        self._checkpointer = checkpointer
        self._conversation_id = conversation_id or uuid4()

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
        self.add_user_message(user_input)

        result, accumulated = await self._executor.run_with_messages(
            list(self._history)
        )
        self._history = accumulated
        await self._truncate_history()

        if self._checkpointer is not None:
            await self._checkpointer.save(
                str(self._conversation_id), list(self._history)
            )

        return result

    async def _truncate_history(self) -> None:
        if self._context_strategy is None:
            return
        self._history = await self._context_strategy.trim(self._history)

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
        model: str = "default",
        generation_config: GenerationConfig | None = None,
        max_iterations: int = 10,
        max_history_turns: int | None = None,
        context_strategy: ContextStrategy | None = None,
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
        )
        saved = await checkpointer.load(str(conversation_id))
        if saved is not None:
            agent._history = list(saved)
        return agent
