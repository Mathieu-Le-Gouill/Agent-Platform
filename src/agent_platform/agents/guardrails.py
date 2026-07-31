from __future__ import annotations

from agent_platform.agents.errors import AgentGuardrailError
from agent_platform.core.interfaces.moderation.base import BaseModerationProvider
from agent_platform.core.middleware import Middleware
from agent_platform.core.schemas.message import AssistantMessage, Message, UserMessage

__all__ = [
    "GuardrailContext",
    "Guardrail",
    "OutputNotEmptyGuardrail",
    "ModerationGuardrail",
]


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


class ModerationGuardrail:
    """Flags user input and/or assistant output with a `BaseModerationProvider`.

    Response-schema validation already has its own validate-and-retry path
    (`Agent.think`), so this guardrail covers the other half of the guardrail
    surface: content safety, not shape.
    """

    def __init__(
        self,
        provider: BaseModerationProvider,
        *,
        check_input: bool = True,
        check_output: bool = True,
    ) -> None:
        self._provider = provider
        self._check_input = check_input
        self._check_output = check_output

    async def before(self, ctx: GuardrailContext) -> AssistantMessage | None:
        if not self._check_input:
            return None
        last_user = next(
            (m for m in reversed(ctx.messages) if isinstance(m, UserMessage)), None
        )
        if last_user is None:
            return None
        result = await self._provider.amoderate(last_user.text)
        if result.flagged:
            raise AgentGuardrailError("Input flagged by moderation")
        return None

    async def after(
        self, ctx: GuardrailContext, result: AssistantMessage
    ) -> AssistantMessage:
        if not self._check_output:
            return result
        moderation = await self._provider.amoderate(result.text)
        if moderation.flagged:
            raise AgentGuardrailError("Output flagged by moderation")
        return result
