from langchain_core.messages import AIMessage

from agent_platform.integrations.llm.response import LLMResponse, FinishReason
from agent_platform.models.token import TokenUsage


def from_langchain(response: AIMessage, model: str) -> LLMResponse:
    
    from agent_platform.integrations.llm.message import AssistantMessage, ToolCall
    
    tool_calls = [
        ToolCall(
            id=tc["id"] or "",
            name=tc["name"],
            arguments=tc["args"]
        )
        for tc in (response.tool_calls or [])
    ]

    usage_meta = response.usage_metadata or {}

    if isinstance(response.content, str):
        content = response.content
    else:
        content = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in response.content
        )

    return LLMResponse(
        message=AssistantMessage(content=content, tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=usage_meta.get("input_tokens", 0),
            output_tokens=usage_meta.get("output_tokens", 0),
        ),
        model=model,
        finish_reason=FinishReason.STOP,
    )