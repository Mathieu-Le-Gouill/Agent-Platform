from __future__ import annotations

from agent_platform.core.schemas.config import ProviderConfig


class ClusteringConfig(ProviderConfig):
    n_clusters: int | None = None
    cluster_labels: list[str] | None = None
    random_state: int | None = None
