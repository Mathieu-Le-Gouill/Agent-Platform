from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageParam, TextBlock, ToolUseBlock
from models.prompt import Prompt

from models.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def to_anthropic_message(message: Message) -> MessageParam:
    """Translate a domain Message into an Anthropic MessageParam dict."""
    
    match message:
        case UserMessage():
            return {"role": "user", "content": message.content}

        case AssistantMessage() if message.tool_calls:
            return {
                "role": "assistant",
                "content": [
                    *(  [{"type": "text", "text": message.content}]
                        if message.content else []
                    ),
                    *[
                        {
                            "type": "tool_use",
                            "id":    tc.id,
                            "name":  tc.name,
                            "input": tc.arguments,  # JSON string → Anthropic expects dict
                        }
                        for tc in message.tool_calls
                    ],
                ],
            }

        case AssistantMessage():
            return {"role": "assistant", "content": message.content}

        case ToolMessage():
            return {
                "role": "user",  # Anthropic embeds tool results inside a user turn
                "content": [
                    {
                        "type":        "tool_result",
                        "tool_use_id": message.result.tool_call_id,
                        "content":     message.result.content,
                        "is_error":    message.result.is_error,
                    }
                ],
            }

        case SystemMessage():
            # Anthropic takes system as a top-level param, not in the messages list.
            # Raise so the adapter knows to extract it before calling to_anthropic.
            raise ValueError(
                "SystemMessage must be extracted separately as the `system` parameter. "
                "Use extract_system(prompt) before translating messages."
            )

        case _:
            raise ValueError(f"Unsupported message type: {type(message)}")


def from_anthropic_message(response: AnthropicMessage) -> AssistantMessage | None:
    """Translate an Anthropic response into a domain AssistantMessage."""

    if not response.content:
        return None

    text = ""
    tool_calls: list[ToolCall] = []

    for block in response.content:
        match block:
            case TextBlock():
                text += block.text
            case ToolUseBlock():
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,  # dict — keep as-is or json.dumps
                    )
                )

    return AssistantMessage(content=text, tool_calls=tool_calls)


def extract_system(prompt: Prompt) -> tuple[str | None, list[Message]]:
    """
    Anthropic requires system instructions as a separate top-level field.
    Returns (system_text, remaining_messages) so the adapter can pass them
    to client.messages.create(system=..., messages=...).
    """

    system = prompt.system_prompt()
    messages: list[Message] = [
        m for m in prompt.messages
        if not isinstance(m, SystemMessage)
    ]
    return system, messages