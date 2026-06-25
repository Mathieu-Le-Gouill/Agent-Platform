from __future__ import annotations
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from agent_platform.models.content import Content
from typing import Optional

@dataclass(slots=True)
class Cluster:
    id: UUID = field(default_factory=uuid4)
    label: Optional[str] = None
    items: list[Content] = field(default_factory=list)
    centroid: Optional[list[float]] = None