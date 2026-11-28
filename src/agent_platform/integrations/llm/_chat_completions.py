"""Reading the chat-completions wire format that OpenAI and Hugging Face share.

Both SDKs return objects of the same shape (`choices[0].message`, `usage`,
stream deltas with indexed tool-call fragments) under different classes, so
the readers here take them structurally.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from agent_platform.core.interfaces.llm.response import (
    LLMResponse,
    StreamChunk,
    ToolCallDelta,
)
from agent_platform.core.schemas.message import AssistantMessage, ToolCall
from agent_platform.core.schemas.token import TokenUsage
from agent_platform.integrations.llm._finish_reason import to_finish_reason

__all__ = ["ChatCompletionStream", "from_chat_completion"]


def from_chat_completion(response: Any, model: str) -> LLMResponse:
    message = response.choices[0].message
    tool_calls = [
        # Only function tools are ever sent, so only function calls come back.
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=json.loads(tc.function.arguments),
        )
        for tc in (message.tool_calls or [])
    ]
    usage = response.usage
    return LLMResponse(
        message=AssistantMessage(content=message.content or "", tool_calls=tool_calls),
        usage=TokenUsage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ),
        model=model,
        finish_reason=to_finish_reason(response.choices[0].finish_reason),
        response_id=response.id,
    )


class ChatCompletionStream:
    """Turns stream chunks into `StreamChunk`s, keeping the usage and the raw
    finish reason, which arrive on their own chunks, for the end of stream."""

    def __init__(self) -> None:
        self.usage: TokenUsage | None = None
        self.raw_finish_reason: str | None = None

    def read(self, chunk: Any) -> Iterator[StreamChunk]:
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.finish_reason is not None:
                self.raw_finish_reason = choice.finish_reason
            if choice.delta.content:
                yield StreamChunk(delta=choice.delta.content)
            tool_call_deltas = [
                ToolCallDelta(
                    index=tc.index,
                    id=tc.id,
                    name=tc.function.name if tc.function else None,
                    arguments_delta=tc.function.arguments if tc.function else None,
                )
                for tc in (choice.delta.tool_calls or [])
            ]
            if tool_call_deltas:
                yield StreamChunk(delta="", tool_call_deltas=tool_call_deltas)
        if chunk.usage is not None:
            self.usage = TokenUsage(
                input_tokens=chunk.usage.prompt_tokens,
                output_tokens=chunk.usage.completion_tokens,
            )
