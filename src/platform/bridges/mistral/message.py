
from typing import Union, Any
import json

from mistralai.client.models import (
    AssistantMessage as MistralAssistantMessage,
    SystemMessage    as MistralSystemMessage,
    UserMessage      as MistralUserMessage,
    ToolMessage      as MistralToolMessage,
    ToolCall         as MistralToolCall,
    FunctionCall     as MistralFunctionCall,
)

from providers.llm.message import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)

MistralMessage = Union[MistralSystemMessage, MistralUserMessage, MistralAssistantMessage, MistralToolMessage]


def to_mistral(message: Message) -> MistralMessage:
    match message:
        case SystemMessage():
            return MistralSystemMessage(content=message.content)

        case UserMessage():
            return MistralUserMessage(content=message.content)

        case AssistantMessage() if message.tool_calls:
            return MistralAssistantMessage(
                content=message.content or None,
                tool_calls=[
                    MistralToolCall(
                        id=tc.id,
                        function=MistralFunctionCall(
                            name=tc.name,
                            arguments=tc.arguments,  # JSON string
                        ),
                    )
                    for tc in message.tool_calls
                ],
            )

        case AssistantMessage():
            return MistralAssistantMessage(content=message.content)

        case ToolMessage():
            return MistralToolMessage(
                tool_call_id=message.result.tool_call_id,
                content=message.result.content,
            )

        case _:
            raise ValueError(f"Unsupported message type: {type(message)}")
        
        
def from_mistral(response: MistralAssistantMessage) -> AssistantMessage | None:
    if response is None:
        return None

    tool_calls = [
        ToolCall(
            id=tc.id or "",
            name=tc.function.name,
            arguments=_to_dict(tc.function.arguments),
        )
        for tc in (response.tool_calls or [])
    ]

    return AssistantMessage(
        content=response.content if isinstance(response.content, str) else "",
        tool_calls=tool_calls,
    )



def _to_dict(arguments: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(arguments, str):
        return json.loads(arguments)
    return arguments