from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Optional
from datetime import datetime
from agent_platform.models.language import Language

@dataclass(slots=True)
class Message:
    id: UUID = field(default_factory=uuid4)
    content: str = ""
    user_id: Optional[UUID] = None

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)

    language: Optional[Language] = None
