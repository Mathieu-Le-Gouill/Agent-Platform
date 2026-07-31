import pytest

from agent_platform.agents.errors import AgentGuardrailError
from agent_platform.agents.guardrails import GuardrailContext, OutputNotEmptyGuardrail
from agent_platform.core.schemas.message import AssistantMessage, UserMessage


class TestOutputNotEmptyGuardrail:
    @pytest.mark.asyncio
    async def test_before_never_short_circuits(self):
        guardrail = OutputNotEmptyGuardrail()
        ctx = GuardrailContext(messages=[UserMessage(content="hi")])

        assert await guardrail.before(ctx) is None

    @pytest.mark.asyncio
    async def test_after_passes_through_non_empty_text(self):
        guardrail = OutputNotEmptyGuardrail()
        ctx = GuardrailContext(messages=[])
        msg = AssistantMessage(content="hello")

        result = await guardrail.after(ctx, msg)

        assert result is msg

    @pytest.mark.asyncio
    async def test_after_allows_tool_calls_with_no_text(self):
        from agent_platform.core.schemas.message import ToolCall

        guardrail = OutputNotEmptyGuardrail()
        ctx = GuardrailContext(messages=[])
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="c1", name="tool", arguments={})],
        )

        result = await guardrail.after(ctx, msg)

        assert result is msg

    @pytest.mark.asyncio
    async def test_after_rejects_empty_text_and_no_tool_calls(self):
        guardrail = OutputNotEmptyGuardrail()
        ctx = GuardrailContext(messages=[])
        msg = AssistantMessage(content="")

        with pytest.raises(AgentGuardrailError, match="empty"):
            await guardrail.after(ctx, msg)
