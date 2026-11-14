from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from agent_platform.agents.errors import AgentGuardrailError
from agent_platform.core.interfaces.moderation.base import BaseModerationProvider
from agent_platform.core.middleware import Middleware
from agent_platform.core.observability import GuardrailCheck, observe
from agent_platform.core.schemas.message import AssistantMessage, Message, UserMessage

__all__ = [
    "GuardrailContext",
    "Guardrail",
    "OutputNotEmptyGuardrail",
    "ModerationGuardrail",
    "TracedGuardrail",
]

_STAGE_INPUT = "input"
_STAGE_OUTPUT = "output"


class GuardrailContext:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = messages


Guardrail = Middleware[GuardrailContext, AssistantMessage]


class TracedGuardrail:
    """Wraps a guardrail so each check is a span and each block is counted.

    A wrapper rather than a base class, so any object satisfying `Guardrail`
    (including third-party ones) gets the same trace evidence without
    inheriting from the platform.
    """

    def __init__(self, guardrail: Guardrail) -> None:
        self._guardrail = guardrail
        self._name = type(guardrail).__name__

    async def before(self, ctx: GuardrailContext) -> AssistantMessage | None:
        with self._traced(_STAGE_INPUT):
            return await self._guardrail.before(ctx)

    async def after(
        self, ctx: GuardrailContext, result: AssistantMessage
    ) -> AssistantMessage:
        with self._traced(_STAGE_OUTPUT):
            return await self._guardrail.after(ctx, result)

    @contextmanager
    def _traced(self, stage: str) -> Iterator[None]:
        with observe(GuardrailCheck(guardrail_name=self._name, stage=stage)) as check:
            try:
                yield
            except AgentGuardrailError:
                check.blocked = True
                raise


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
