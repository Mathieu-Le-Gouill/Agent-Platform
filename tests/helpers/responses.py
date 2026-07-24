from agent_platform.core.interfaces.llm.response import FinishReason, LLMResponse
from agent_platform.core.schemas.message import AssistantMessage, ToolCall
from agent_platform.core.schemas.token import TokenUsage


def make_fake_llm_response(
    content: str = "", tool_calls: list | None = None
) -> LLMResponse:
    return LLMResponse(
        message=AssistantMessage(
            content=content,
            tool_calls=[
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["args"])
                for tc in (tool_calls or [])
            ],
        ),
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        model="test-model",
        finish_reason=FinishReason.STOP,
    )
