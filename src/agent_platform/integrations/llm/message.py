from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Union, Any
from datetime import datetime
from enum import Enum

from agent_platform.models.enums.language import Language


# --- Roles ---

class MessageRole(Enum):
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"
    TOOL      = "tool"


# --- Tool call ---

@dataclass(slots=True, frozen=True)
class ToolCall:
    """Represents a single tool invocation requested by the assistant"""
    id:        str  # provider's tool-call id  (e.g. "call_abc123")
    name:      str  # function/tool name
    arguments: dict[str, Any]  # structured arguments, serialisation is the adapter's job


@dataclass(slots=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution, linked to a ToolCall"""
    tool_call_id: str     # must match ToolCall.id
    name:         str     # echoed for readability
    content:      str     # serialised result (JSON, plain text, error message…)
    is_error:     bool = False


# --- Base class ---

@dataclass(slots=True)
class BaseMessage:
    """
    Common fields shared by every message variant
    Never instantiated directly, use the concrete subclasses below
    """
    role:       MessageRole
    id:         UUID     = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)
    language:   Language | None = None

    # Provider-level metadata (model name, token counts, finish reason…)
    # Kept as a plain dict so callers can store whatever a provider returns
    # without forcing a schema change here.
    metadata: dict = field(default_factory=dict)


# --- Child classes ---

@dataclass(slots=True)
class SystemMessage(BaseMessage):
    """
    High-priority instructions that frame the conversation
    Usually the first message in a prompt, invisible to the end-user
    """
    role:    MessageRole = field(default=MessageRole.SYSTEM, init=False)
    content: str = ""


@dataclass(slots=True)
class UserMessage(BaseMessage):
    """
    A turn from the human side of the conversation
    user_id lets you track multi-user sessions
    """
    role:    MessageRole = field(default=MessageRole.USER, init=False)
    content: str = ""
    user_id: UUID | None = None


@dataclass(slots=True)
class AssistantMessage(BaseMessage):
    """
    A turn produced by the LLM
    tool_calls is populated when the model requests tool execution
    instead of (or in addition to) producing text
    """
    role:       MessageRole       = field(default=MessageRole.ASSISTANT, init=False)
    content:    str               = ""
    tool_calls: list[ToolCall]    = field(default_factory=list)


@dataclass(slots=True)
class ToolMessage(BaseMessage):
    """
    The result of a tool execution, sent back to the model
    Must be paired with an AssistantMessage that contains the matching ToolCall
    """
    role:   MessageRole = field(default=MessageRole.TOOL, init=False)
    result: ToolResult  = field(default_factory=lambda: ToolResult("", "", ""))


# --- Union type ---

Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]