from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass(slots=True)
class Content:
    id: UUID = field(default_factory=uuid4)
    content: str = ""