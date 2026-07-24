from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    StreamChunk,
)
from agent_platform.core.schemas.message import AssistantMessage, ToolCall
from agent_platform.core.schemas.token import TokenUsage


class TestFinishReason:
    def test_values(self):
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.STOP_SEQUENCE.value == "stop_sequence"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.TOOL_CALL.value == "tool_call"
        assert FinishReason.CONTENT_FILTER.value == "content_filter"
        assert FinishReason.ERROR.value == "error"
        assert FinishReason.UNKNOWN.value == "unknown"

    def test_is_str_enum(self):
        assert issubclass(FinishReason, str)


class TestLLMResponse:
    def test_minimal(self):
        msg = AssistantMessage(content="Hello")
        usage = TokenUsage(input_tokens=5, output_tokens=10)
        response = LLMResponse(message=msg, usage=usage, model="gpt-4")
        assert response.message is msg
        assert response.usage is usage
        assert response.model == "gpt-4"
        assert response.latency_ms is None
        assert response.finish_reason == FinishReason.STOP

    def test_full(self):
        msg = AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="c1", name="f", arguments={"a": 1})],
        )
        usage = TokenUsage(input_tokens=1, output_tokens=2)
        response = LLMResponse(
            message=msg,
            usage=usage,
            model="claude-3",
            latency_ms=150.5,
            finish_reason=FinishReason.TOOL_CALL,
        )
        assert response.latency_ms == 150.5
        assert response.finish_reason == FinishReason.TOOL_CALL

    def test_message_none(self):
        usage = TokenUsage.zero()
        response = LLMResponse(message=None, usage=usage, model="test")
        assert response.message is None

    def test_frozen(self):
        import pytest
        from pydantic import ValidationError

        usage = TokenUsage.zero()
        response = LLMResponse(
            message=AssistantMessage(content="hi"),
            usage=usage,
            model="test",
        )
        with pytest.raises(ValidationError):
            response.model = "other"


class TestStreamChunk:
    def test_minimal(self):
        chunk = StreamChunk(delta="Hello")
        assert chunk.delta == "Hello"
        assert chunk.finish_reason == FinishReason.STOP
        assert chunk.usage is None

    def test_with_finish_reason(self):
        chunk = StreamChunk(delta="", finish_reason=FinishReason.STOP)
        assert chunk.finish_reason == FinishReason.STOP

    def test_with_usage(self):
        usage = TokenUsage(input_tokens=10, output_tokens=20)
        chunk = StreamChunk(delta="done", finish_reason=FinishReason.STOP, usage=usage)
        assert chunk.usage is usage
        assert chunk.usage.input_tokens == 10

    def test_frozen(self):
        import pytest
        from pydantic import ValidationError

        chunk = StreamChunk(delta="test")
        with pytest.raises(ValidationError):
            chunk.delta = "changed"
