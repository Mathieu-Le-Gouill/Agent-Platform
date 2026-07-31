from collections.abc import AsyncIterator

from agent_platform.core.interfaces.llm.response import (
    FinishReason,
    LLMResponse,
    StreamChunk,
    ToolCallDelta,
)
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


def make_text_stream_chunks(text_parts: list[str]) -> list[StreamChunk]:
    chunks = [StreamChunk(delta=part) for part in text_parts]
    chunks.append(StreamChunk(delta="", finish_reason=FinishReason.STOP))
    return chunks


def make_tool_call_stream_chunks(tool_calls: list[dict]) -> list[StreamChunk]:
    import json

    deltas = [
        ToolCallDelta(
            index=i,
            id=tc["id"],
            name=tc["name"],
            arguments_delta=json.dumps(tc["args"]),
        )
        for i, tc in enumerate(tool_calls)
    ]
    return [
        StreamChunk(delta="", tool_call_deltas=deltas),
        StreamChunk(delta="", finish_reason=FinishReason.TOOL_CALL),
    ]


async def make_fake_stream(chunks: list[StreamChunk]) -> AsyncIterator[StreamChunk]:
    for chunk in chunks:
        yield chunk
