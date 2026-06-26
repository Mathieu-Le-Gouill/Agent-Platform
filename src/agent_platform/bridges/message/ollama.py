from typing import Literal, TypedDict, Any

from models.message import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)


class OllamaMessage(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[dict[str, Any]]
    tool_call_id: str


def to_ollama(message: Message) -> OllamaMessage:
    match message:
        case SystemMessage():
            return {
                "role": "system",
                "content": message.content,
            }

        case UserMessage():
            return {
                "role": "user",
                "content": message.content,
            }

        case AssistantMessage() if message.tool_calls:
            return {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ],
            }

        case AssistantMessage():
            return {
                "role": "assistant",
                "content": message.content or "",
            }

        case ToolMessage():
            return {
                "role": "tool",
                "tool_call_id": message.result.tool_call_id,
                "content": message.result.content,
            }

        case _:
            raise ValueError(...)


def from_ollama(response: OllamaMessage) -> AssistantMessage | None:
    if response is None:
        return None

    tool_calls = []

    for i, tc in enumerate(response.get("tool_calls", [])):
        function = tc.get("function", {})

        tool_calls.append(
            ToolCall(
                id=str(i),
                name=function["name"],
                arguments=function.get("arguments", {}),
            )
        )

    return AssistantMessage(
        content=response.get("content", ""),
        tool_calls=tool_calls,
    )