from __future__ import annotations

from dataclasses import dataclass, field

from agent_platform.integrations.llm.message import Message, SystemMessage, ToolCall, ToolMessage, ToolResult, UserMessage, AssistantMessage
from agent_platform.models.enums.language import Language


@dataclass(slots=True)
class Prompt:
    """
    A complete, ordered list of messages ready to be sent to any LLM provider.

    Usage
    -----
        prompt = Prompt.build(
            system="You are a concise assistant.",
            history=[...],
            user="Summarise this document.",
        )
    """
    messages: list[Message] = field(default_factory=list)

    # --- Constructors ---

    @classmethod
    def build(
        cls,
        *,
        system: str | None = None,
        history: list[Message] | None = None,
        user: str | None = None,
        language: Language | None = None,
    ) -> Prompt:
        """
        Canonical factory used by pipelines and nodes.

            system  → prepended SystemMessage (skipped if None)
            history → injected as-is (prior conversation turns)
            user    → appended UserMessage (skipped if None)
        """
        messages: list[Message] = []

        if system:
            messages.append(SystemMessage(content=system, language=language))

        if history:
            messages.extend(history)

        if user:
            messages.append(UserMessage(content=user, language=language))

        return cls(messages=messages)


    # --- Helpers ---

    def add_system(self, content: str, **kw) -> Prompt:
        self.messages.append(SystemMessage(content=content, **kw))
        return self


    def add_user(self, content: str, **kw) -> Prompt:
        self.messages.append(UserMessage(content=content, **kw))
        return self


    def add_assistant(self, content: str, tool_calls: list[ToolCall] | None = None, **kw) -> Prompt:
        self.messages.append(AssistantMessage(content=content, tool_calls=tool_calls or [], **kw))
        return self


    def add_tool_result(self, result: ToolResult, **kw) -> Prompt:
        self.messages.append(ToolMessage(result=result, **kw))
        return self


    # --- Introspection ---

    def last_user_message(self) -> UserMessage | None:
        return next(
            (m for m in reversed(self.messages) if isinstance(m, UserMessage)),
            None,
        )


    def system_prompt(self) -> str | None:
        msg = next((m for m in self.messages if isinstance(m, SystemMessage)), None)
        return msg.content if msg else None