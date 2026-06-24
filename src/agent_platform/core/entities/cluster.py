from __future__ import annotations
from dataclasses import dataclass, field
from uuid import UUID, uuid4
   

@dataclass(slots=True)
class Cluster:
    id: UUID = field(default_factory=uuid4)