from __future__ import annotations

import os
from enum import StrEnum

from pydantic import BaseModel

__all__ = ["TracingBackend", "TracingConfig"]


class TracingBackend(StrEnum):
    NONE = "none"
    LANGSMITH = "langsmith"
    LANGFUSE = "langfuse"


class TracingConfig(BaseModel):
    backend: TracingBackend = TracingBackend.NONE

    @classmethod
    def from_env(cls) -> TracingConfig:
        raw = os.getenv("AGENT_PLATFORM_TRACING", "none").lower()
        try:
            backend = TracingBackend(raw)
        except ValueError:
            backend = TracingBackend.NONE
        return cls(backend=backend)
