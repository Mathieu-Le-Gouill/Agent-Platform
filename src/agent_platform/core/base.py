from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Entity(BaseModel, frozen=True):
    id: UUID = Field(default_factory=uuid4)


class Timestamped(BaseModel, frozen=True):
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime | None = None
