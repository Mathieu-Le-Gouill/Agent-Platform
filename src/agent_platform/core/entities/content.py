from dataclasses import dataclass

@dataclass(slots=True)
class Content:
    id: str
    content: str