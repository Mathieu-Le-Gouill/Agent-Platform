from __future__ import annotations

from uuid import UUID, uuid4

from agent_platform.agents.agent import Agent
from agent_platform.agents.executor import AgentExecutor
from agent_platform.agents.tools.registry import ToolRegistry
from agent_platform.core.interfaces.llm.base import BaseLLMProvider
from agent_platform.core.interfaces.llm.config import GenerationConfig
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
        conversation_id: UUID | None = None,
    ) -> None:
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
        self._max_history_turns = max_history_turns
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
        self._truncate_history()

        return result

    def _truncate_history(self) -> None:
        if self._max_history_turns is None:
            return

        user_indices = [
            i for i, m in enumerate(self._history) if isinstance(m, UserMessage)
        ]
        if len(user_indices) <= self._max_history_turns:
            return

        first_to_keep = user_indices[-self._max_history_turns]
        self._history = self._history[first_to_keep:]
