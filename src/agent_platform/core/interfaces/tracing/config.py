from __future__ import annotations

from pydantic import BaseModel

__all__ = ["TracingConfig"]


class TracingConfig(BaseModel):
    """Settings every tracing backend understands.

    Backend-specific knobs (exporter protocol, endpoints, credentials) belong
    on a subclass owned by that backend's integration, along with the
    `from_env()` that reads them, since the env vars a backend honors are part
    of that backend, not of this contract.
    """

    service_name: str = "agent-platform"
