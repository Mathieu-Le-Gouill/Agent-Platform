from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from models.protocols.text_unit import TextUnit


@dataclass(slots=True)
class Cluster:
    id: UUID = field(default_factory=uuid4)
    label: str | None = None
    items: list[TextUnit] = field(default_factory=list)
    centroid: list[float] | None = None