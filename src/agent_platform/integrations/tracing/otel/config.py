from __future__ import annotations

import os
from enum import StrEnum

from agent_platform.core.interfaces.tracing.config import TracingConfig

__all__ = ["OTelBackend", "OTelTracingConfig"]


class OTelBackend(StrEnum):
    AUTO = "auto"
    NONE = "none"
    CONSOLE = "console"


class OTelTracingConfig(TracingConfig):
    """How the OpenTelemetry tracer exports spans.

    `AUTO` exports over OTLP whenever the standard endpoint env var is set and
    stays a no-op otherwise, `NONE` force-disables, `CONSOLE` prints to stdout.
    """

    backend: OTelBackend = OTelBackend.AUTO

    @classmethod
    def from_env(cls) -> OTelTracingConfig:
        raw = os.getenv("AGENT_PLATFORM_TRACING", "auto").lower()
        try:
            backend = OTelBackend(raw)
        except ValueError:
            backend = OTelBackend.AUTO
        service_name = os.getenv(
            "AGENT_PLATFORM_SERVICE_NAME",
            os.getenv("OTEL_SERVICE_NAME", "agent-platform"),
        )
        return cls(backend=backend, service_name=service_name)
