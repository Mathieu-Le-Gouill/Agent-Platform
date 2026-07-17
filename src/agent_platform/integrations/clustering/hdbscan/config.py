from __future__ import annotations
from agent_platform.core.interfaces.clustering.config import ClusteringConfig


class HDBSCANConfig(ClusteringConfig):
    min_cluster_size: int = 5
    min_samples: int | None = None
    cluster_selection_epsilon: float = 0.0
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
