from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Optional
from agent_platform.core.entities.content import Content
from datetime import datetime
from core.value_objects.language import Language

@dataclass(slots=True)
class Message(Content):
    id: UUID = field(default_factory=uuid4)
    content: str = ""
    user_id: Optional[UUID] = None

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)

    language: Optional[Language] = None
