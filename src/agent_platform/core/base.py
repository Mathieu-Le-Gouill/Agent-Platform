from uuid import UUID, uuid4
from datetime import datetime

from pydantic import BaseModel, Field


class Entity(BaseModel):
    id: UUID = Field(default_factory=uuid4)


class Timestamped(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime | None = None
