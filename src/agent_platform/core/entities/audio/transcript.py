from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Optional
from agent_platform.core.entities.content import Content

@dataclass(slots=True)
class Transcript(Content):
    id: UUID = field(default_factory=uuid4)
    content: str = ""
    audio_segment_id: Optional[UUID] = None

    # Must find the differents users like a conversation ? Maybe with messages ?
