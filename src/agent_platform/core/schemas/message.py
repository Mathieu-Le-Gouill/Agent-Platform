from __future__ import annotations

from uuid import UUID, uuid4
from typing import Union, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from agent_platform.core.schemas.enums import Language


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel, frozen=True):
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel, frozen=True):
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


class BaseMessage(BaseModel):
    role: MessageRole
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    language: Language | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SystemMessage(BaseMessage):
    role: MessageRole = MessageRole.SYSTEM
    content: str = ""


class UserMessage(BaseMessage):
    role: MessageRole = MessageRole.USER
    content: str = ""
    user_id: UUID | None = None


class AssistantMessage(BaseMessage):
    role: MessageRole = MessageRole.ASSISTANT
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolMessage(BaseMessage):
    role: MessageRole = MessageRole.TOOL
    result: ToolResult = Field(
        default_factory=lambda: ToolResult(tool_call_id="", name="", content="")
    )


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]


class Prompt(BaseModel):
    messages: list[Message] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        system: str | None = None,
        history: list[Message] | None = None,
        user: str | None = None,
        language: Language | None = None,
    ) -> Prompt:
        messages: list[Message] = []

        if system:
            messages.append(SystemMessage(content=system, language=language))

        if history:
            messages.extend(history)

        if user:
            messages.append(UserMessage(content=user, language=language))

        return cls(messages=messages)

    def add_system(self, content: str, **kw) -> Prompt:
        self.messages.append(SystemMessage(content=content, **kw))
        return self

    def add_user(self, content: str, **kw) -> Prompt:
        self.messages.append(UserMessage(content=content, **kw))
        return self

    def add_assistant(
        self, content: str, tool_calls: list[ToolCall] | None = None, **kw
    ) -> Prompt:
        self.messages.append(
            AssistantMessage(content=content, tool_calls=tool_calls or [], **kw)
        )
        return self

    def add_tool_result(self, result: ToolResult, **kw) -> Prompt:
        self.messages.append(ToolMessage(result=result, **kw))
        return self

    def last_user_message(self) -> UserMessage | None:
        return next(
            (m for m in reversed(self.messages) if isinstance(m, UserMessage)),
            None,
        )

    def system_prompt(self) -> str | None:
        msg = next((m for m in self.messages if isinstance(m, SystemMessage)), None)
        return msg.content if msg else None
