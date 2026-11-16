from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from uuid import UUID, uuid4

from agent_platform.agents.conversation import ConversationAgent
from agent_platform.core.errors import NotFoundError


class ConversationNotFound(NotFoundError):
    """Raised for an unknown conversation id and for another key's id alike,
    so a caller cannot tell whether a conversation it does not own exists.
    """


@dataclass(frozen=True)
class Conversation:
    """A conversation's agent and the lock that serializes its turns.

    A turn that already holds this object keeps working after the store evicts
    the conversation; the agent and lock are freed once the turn drops it.
    """

    id: UUID
    agent: ConversationAgent
    lock: asyncio.Lock


class ConversationStore:
    """In-memory conversations, private to the key that created them.

    Each key keeps at most `max_per_key` conversations; creating one more
    evicts that key's least recently used one. A key never affects another
    key's conversations. State is per process and lost on restart.
    """

    def __init__(self, template: ConversationAgent, max_per_key: int) -> None:
        if max_per_key < 1:
            raise ValueError("max_per_key must be >= 1")
        self._template = template
        self._max_per_key = max_per_key
        self._by_key: dict[str, OrderedDict[UUID, Conversation]] = {}

    def create(self, key_id: str) -> Conversation:
        """Starts a conversation with a server-generated id for `key_id`."""
        conversation_id = uuid4()
        conversation = Conversation(
            id=conversation_id,
            agent=self._template.with_conversation(conversation_id),
            lock=asyncio.Lock(),
        )
        owned = self._by_key.setdefault(key_id, OrderedDict())
        owned[conversation_id] = conversation
        while len(owned) > self._max_per_key:
            owned.popitem(last=False)
        return conversation

    def get(self, key_id: str, conversation_id: UUID) -> Conversation:
        """Returns the conversation and marks it most recently used.

        Raises:
            ConversationNotFound: the id is unknown or belongs to another key.
        """
        owned = self._by_key.get(key_id)
        if owned is None or conversation_id not in owned:
            raise ConversationNotFound("conversation not found")
        owned.move_to_end(conversation_id)
        return owned[conversation_id]


__all__ = ["Conversation", "ConversationNotFound", "ConversationStore"]
