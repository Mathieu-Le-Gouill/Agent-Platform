from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDERS: dict[str, str] = {
    "HDBSCANClusterer": "agent_platform.integrations.clustering.hdbscan.hdbscan",
    "KMeansClusterer": "agent_platform.integrations.clustering.kmeans.kmeans",
    "GMMClusterer": "agent_platform.integrations.clustering.gmm.gmm",
}


def __getattr__(name: str) -> Any:
    if name in _PROVIDERS:
        return getattr(import_module(_PROVIDERS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_PROVIDERS.keys()))
