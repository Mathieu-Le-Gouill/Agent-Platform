from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4
from typing import Optional

from models.protocols.text_unit import TextUnit


@dataclass(slots=True)
class Cluster:
    id: UUID = field(default_factory=uuid4)
    label: Optional[str] = None
    items: list[TextUnit] = field(default_factory=list)
    centroid: Optional[list[float]] = None