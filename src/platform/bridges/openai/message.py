from typing import Any
import json

from providers.llm.message import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageParam,
)


def to_openai(message: Message) -> ChatCompletionMessageParam:
    match message:
        case SystemMessage():
            return ChatCompletionSystemMessageParam(
                role="system",
                content=message.content,
            )

        case UserMessage():
            return ChatCompletionUserMessageParam(
                role="user",
                content=message.content,
            )

        case AssistantMessage():
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=message.content,
            )

        case ToolMessage():
            return ChatCompletionToolMessageParam(
                role="tool",
                tool_call_id=message.result.tool_call_id,
                content=message.result.content,
            )

        case _:
            raise ValueError(...)
        

def from_openai(response) -> AssistantMessage | None:
    if response is None:
        return None

    tool_calls = [
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=_to_dict(tc.function.arguments),
        )
        for tc in (response.tool_calls or [])
    ]

    return AssistantMessage(
        content=response.content or "",
        tool_calls=tool_calls,
    )


def _to_dict(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, str):
        return json.loads(arguments)

    return arguments