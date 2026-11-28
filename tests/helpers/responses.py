from collections.abc import AsyncIterator
from types import SimpleNamespace

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


# Native chat-completions shapes, as the OpenAI-compatible SDKs (OpenAI,
# Hugging Face) return them.


def make_chat_tool_call(id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def make_chat_completion_chunk(
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def make_chat_completion(
    content: str | None,
    tool_calls: list[SimpleNamespace] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    finish_reason: str | None = "stop",
    id: str = "resp-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )
