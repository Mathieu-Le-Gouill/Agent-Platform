from __future__ import annotations

from agent_platform.agents.errors import AgentGuardrailError
from agent_platform.core.middleware import Middleware
from agent_platform.core.schemas.message import AssistantMessage, Message

__all__ = ["GuardrailContext", "Guardrail", "OutputNotEmptyGuardrail"]


class GuardrailContext:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages


Guardrail = Middleware[GuardrailContext, AssistantMessage]


class OutputNotEmptyGuardrail:
    """Rejects a final assistant answer that has no text and no tool calls."""

    async def before(self, ctx: GuardrailContext) -> AssistantMessage | None:
        return None

    async def after(
        self, ctx: GuardrailContext, result: AssistantMessage
    ) -> AssistantMessage:
        if not result.tool_calls and not result.text.strip():
            raise AgentGuardrailError("Guardrail rejected an empty assistant response")
        return result
