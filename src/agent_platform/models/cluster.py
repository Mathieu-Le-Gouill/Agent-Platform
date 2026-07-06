from __future__ import annotations

from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from agent_platform.models.document import TextDocument
from agent_platform.models.chunk import TextChunk


class Cluster(BaseModel, frozen=True):
    id: UUID = Field(default_factory=uuid4)
    label: str = ""
    items: list[TextChunk | TextDocument] = Field(default_factory=list)
    centroid: list[float] | None = None
