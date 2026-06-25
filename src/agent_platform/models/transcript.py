from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Optional

from models.language import Language

@dataclass(slots=True)
class Transcript:
    id: UUID = field(default_factory=uuid4)
    text: str = ""
    audio_segment_id: Optional[UUID] = None
    language: Optional[Language] = None

    # Must find the differents users like a conversation ? Maybe with messages ?
