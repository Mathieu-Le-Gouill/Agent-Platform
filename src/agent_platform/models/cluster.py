from __future__ import annotations
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from models.protocols.text_unit import TextUnit
from typing import Optional

@dataclass(slots=True)
class Cluster:
    id: UUID = field(default_factory=uuid4)
    label: Optional[str] = None
    items: list[TextUnit] = field(default_factory=list)
    centroid: Optional[list[float]] = None