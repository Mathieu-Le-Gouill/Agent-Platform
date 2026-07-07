from __future__ import annotations

from agent_platform.agents.agent import Agent
from agent_platform.core.errors import AgentMaxIterations
from agent_platform.models.message import Message, UserMessage


class AgentExecutor:
    def __init__(
        self,
        agent: Agent,
        max_iterations: int = 10,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._agent = agent
        self._max_iterations = max_iterations

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    async def run(self, user_input: str) -> str:
        messages: list[Message] = [UserMessage(content=user_input)]
        result, _ = await self._execute(messages)
        return result

    async def run_with_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[Message]]:
        return await self._execute(list(messages))

    async def _execute(
        self, messages: list[Message]
    ) -> tuple[str, list[Message]]:
        for _ in range(self._max_iterations):
            assistant_msg = await self._agent.think(messages)
            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                return assistant_msg.content, messages

            tool_messages = await self._agent.act(assistant_msg)
            messages.extend(tool_messages)

        raise AgentMaxIterations(
            f"Agent '{self._agent.name}' exceeded "
            f"max iterations ({self._max_iterations})"
        )
