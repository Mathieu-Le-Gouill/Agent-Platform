from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from models.language import Language

@dataclass(slots=True)
class Transcript:
    id: UUID = field(default_factory=uuid4)
    text: str = ""
    audio_segment_id: UUID | None = None
    language: Language | None = None

    # Must find the differents users like a conversation ? Maybe with messages ?
