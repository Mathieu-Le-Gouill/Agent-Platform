from __future__ import annotations

from agent_platform.utils.lazy_imports import make_lazy_provider_accessors

_PROVIDERS: dict[str, str] = {
    "HDBSCANClusterer": "agent_platform.integrations.clustering.hdbscan.provider",
    "KMeansClusterer": "agent_platform.integrations.clustering.kmeans.provider",
    "GMMClusterer": "agent_platform.integrations.clustering.gmm.provider",
}

__getattr__, __dir__ = make_lazy_provider_accessors(_PROVIDERS, globals())
