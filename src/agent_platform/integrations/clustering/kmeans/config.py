from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.clustering.config import ClusteringConfig


class KMeansConfig(ClusteringConfig):
    n_clusters: int = 8
    init: str = "k-means++"
    n_init: int | Literal["auto", "warn"] = 10
    max_iter: int = 300
    tol: float = 1e-4
    algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd"
    random_state: int | None = None
