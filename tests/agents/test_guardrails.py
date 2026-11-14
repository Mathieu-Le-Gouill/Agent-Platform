from unittest.mock import AsyncMock

import pytest

from agent_platform.agents.errors import AgentGuardrailError
from agent_platform.agents.guardrails import (
    GuardrailContext,
    ModerationGuardrail,
    OutputNotEmptyGuardrail,
    TracedGuardrail,
)
from agent_platform.core.interfaces.moderation.response import ModerationResult
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


def _fake_provider(*, flagged: bool) -> AsyncMock:
    provider = AsyncMock()
    provider.amoderate.return_value = ModerationResult(flagged=flagged, categories=[])
    return provider


class TestModerationGuardrail:
    @pytest.mark.asyncio
    async def test_before_flags_last_user_message(self):
        provider = _fake_provider(flagged=True)
        guardrail = ModerationGuardrail(provider)
        ctx = GuardrailContext(messages=[UserMessage(content="bad input")])

        with pytest.raises(AgentGuardrailError, match="Input flagged"):
            await guardrail.before(ctx)

        provider.amoderate.assert_awaited_once_with("bad input")

    @pytest.mark.asyncio
    async def test_before_passes_clean_input(self):
        provider = _fake_provider(flagged=False)
        guardrail = ModerationGuardrail(provider)
        ctx = GuardrailContext(messages=[UserMessage(content="fine")])

        assert await guardrail.before(ctx) is None

    @pytest.mark.asyncio
    async def test_before_skipped_when_check_input_false(self):
        provider = _fake_provider(flagged=True)
        guardrail = ModerationGuardrail(provider, check_input=False)
        ctx = GuardrailContext(messages=[UserMessage(content="bad input")])

        assert await guardrail.before(ctx) is None
        provider.amoderate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_before_no_user_message_returns_none(self):
        provider = _fake_provider(flagged=True)
        guardrail = ModerationGuardrail(provider)
        ctx = GuardrailContext(messages=[])

        assert await guardrail.before(ctx) is None
        provider.amoderate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_after_flags_output(self):
        provider = _fake_provider(flagged=True)
        guardrail = ModerationGuardrail(provider)
        ctx = GuardrailContext(messages=[])
        msg = AssistantMessage(content="bad output")

        with pytest.raises(AgentGuardrailError, match="Output flagged"):
            await guardrail.after(ctx, msg)

    @pytest.mark.asyncio
    async def test_after_passes_clean_output(self):
        provider = _fake_provider(flagged=False)
        guardrail = ModerationGuardrail(provider)
        ctx = GuardrailContext(messages=[])
        msg = AssistantMessage(content="fine")

        result = await guardrail.after(ctx, msg)

        assert result is msg

    @pytest.mark.asyncio
    async def test_after_skipped_when_check_output_false(self):
        provider = _fake_provider(flagged=True)
        guardrail = ModerationGuardrail(provider, check_output=False)
        ctx = GuardrailContext(messages=[])
        msg = AssistantMessage(content="bad output")

        result = await guardrail.after(ctx, msg)

        assert result is msg
        provider.amoderate.assert_not_awaited()


class TestTracedGuardrail:
    async def test_a_passing_check_is_traced_as_not_blocked(self, recorded_spans):
        guardrail = TracedGuardrail(OutputNotEmptyGuardrail())
        ctx = GuardrailContext(messages=[])

        await guardrail.after(ctx, AssistantMessage(content="fine"))

        (span,) = recorded_spans.get_finished_spans()
        assert span.name == "guardrail"
        assert span.attributes["guardrail.name"] == ("OutputNotEmptyGuardrail")
        assert span.attributes["guardrail.stage"] == "output"
        assert span.attributes["guardrail.blocked"] is False

    async def test_the_input_stage_is_traced_too(self, recorded_spans):
        guardrail = TracedGuardrail(OutputNotEmptyGuardrail())

        await guardrail.before(GuardrailContext(messages=[]))

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["guardrail.stage"] == "input"

    async def test_a_block_is_recorded(self, recorded_spans):
        guardrail = TracedGuardrail(OutputNotEmptyGuardrail())

        with pytest.raises(AgentGuardrailError):
            await guardrail.after(
                GuardrailContext(messages=[]), AssistantMessage(content="")
            )

        (span,) = recorded_spans.get_finished_spans()
        assert span.attributes["guardrail.blocked"] is True
        assert span.attributes["guardrail.stage"] == "output"

    async def test_agent_guardrails_are_traced_without_extra_wiring(
        self, mock_llm, recorded_spans
    ):
        from agent_platform.agents.agent import Agent
        from tests.helpers import make_fake_llm_response

        mock_llm.agenerate.return_value = make_fake_llm_response(content="hi")
        agent = Agent(name="test", llm=mock_llm, guardrails=[OutputNotEmptyGuardrail()])

        await agent.think([UserMessage(content="hi")])

        names = [s.name for s in recorded_spans.get_finished_spans()]
        assert names.count("guardrail") == 2
