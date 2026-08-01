from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agent_platform.core.schemas.document import AudioDocument, ImageDocument
from agent_platform.core.schemas.enums import Language, MediaType

# --- Roles ---


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# --- Tool calls ---


class ToolCall(BaseModel, frozen=True):
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel, frozen=True):
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


# --- Content blocks ---


class TextBlock(BaseModel, frozen=True):
    media_type: Literal[MediaType.TEXT] = MediaType.TEXT
    text: str = ""


class ImageBlock(BaseModel, frozen=True):
    media_type: Literal[MediaType.IMAGE] = MediaType.IMAGE
    image: ImageDocument | str  # str = URL


class AudioBlock(BaseModel, frozen=True):
    media_type: Literal[MediaType.AUDIO] = MediaType.AUDIO
    audio: AudioDocument


ContentBlock = Annotated[
    TextBlock | ImageBlock | AudioBlock, Field(discriminator="media_type")
]


# --- Messages ---


class BaseMessage(BaseModel):
    role: MessageRole
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    language: Language | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContentMessage(BaseMessage):
    content: str | list[ContentBlock] = ""

    @property
    def blocks(self) -> list[ContentBlock]:
        if isinstance(self.content, str):
            return [TextBlock(text=self.content)] if self.content else []
        return self.content

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.blocks if isinstance(b, TextBlock))


class SystemMessage(ContentMessage):
    role: MessageRole = MessageRole.SYSTEM


class UserMessage(ContentMessage):
    role: MessageRole = MessageRole.USER
    user_id: UUID | None = None


class AssistantMessage(ContentMessage):
    role: MessageRole = MessageRole.ASSISTANT
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolMessage(BaseMessage):
    role: MessageRole = MessageRole.TOOL
    result: ToolResult = Field(
        default_factory=lambda: ToolResult(tool_call_id="", name="", content="")
    )


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


# --- Prompt ---


class Prompt(
    BaseModel
):  # not frozen, unlike other schemas: add_*() mutate messages in place
    messages: list[Message] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        system: str | None = None,
        history: list[Message] | None = None,
        user: str | None = None,
        user_blocks: list[ContentBlock] | None = None,
        language: Language | None = None,
    ) -> Prompt:
        messages: list[Message] = []

        if system:
            messages.append(SystemMessage(content=system, language=language))

        if history:
            messages.extend(history)

        if user_blocks:
            messages.append(UserMessage(content=user_blocks, language=language))
        elif user:
            messages.append(UserMessage(content=user, language=language))

        return cls(messages=messages)

    def add_system(self, content: str, **kw: Any) -> Prompt:
        self.messages.append(SystemMessage(content=content, **kw))
        return self

    def add_user(self, content: str, **kw: Any) -> Prompt:
        self.messages.append(UserMessage(content=content, **kw))
        return self

    def add_user_content(self, blocks: list[ContentBlock], **kw: Any) -> Prompt:
        self.messages.append(UserMessage(content=blocks, **kw))
        return self

    def add_assistant(
        self, content: str, tool_calls: list[ToolCall] | None = None, **kw: Any
    ) -> Prompt:
        self.messages.append(
            AssistantMessage(content=content, tool_calls=tool_calls or [], **kw)
        )
        return self

    def add_tool_result(self, result: ToolResult, **kw: Any) -> Prompt:
        self.messages.append(ToolMessage(result=result, **kw))
        return self

    def last_user_message(self) -> UserMessage | None:
        return next(
            (m for m in reversed(self.messages) if isinstance(m, UserMessage)),
            None,
        )

    def system_prompt(self) -> str | None:
        msg = next((m for m in self.messages if isinstance(m, SystemMessage)), None)
        return msg.text if msg else None
