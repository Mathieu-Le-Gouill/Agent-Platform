from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage as LCSystemMessage,
    AIMessage, ToolMessage as LCToolMessage
)
from integrations.llm.message import (
    SystemMessage, UserMessage, AssistantMessage, ToolMessage
)
from integrations.llm.prompt import Prompt


def to_langchain(prompt: Prompt) -> list[BaseMessage]:
    result = []
    for m in prompt.messages:
        match m:
            case SystemMessage(): 
                result.append(LCSystemMessage(content=m.content))

            case UserMessage(): 
                result.append(HumanMessage(content=m.content))

            case AssistantMessage(): 
                result.append(AIMessage(content=m.content))

            case ToolMessage():
                result.append(LCToolMessage(
                    content=m.result.content,
                    tool_call_id=m.result.tool_call_id,
                ))
    return result


