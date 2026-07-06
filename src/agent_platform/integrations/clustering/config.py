from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True, frozen=True)
class ClusteringConfig:
    n_clusters: int | None = None
    cluster_labels: list[str] | None = None
    random_state: int | None = None


@dataclass(slots=True, frozen=True)
class HDBSCANConfig(ClusteringConfig):
    min_cluster_size: int = 5
    min_samples: int | None = None
    metric: str = "euclidean"
    cluster_selection_epsilon: float = 0.0


@dataclass(slots=True, frozen=True)
class KMeansConfig(ClusteringConfig):
    n_init: int | Literal["auto", "warn"] = "auto"
    max_iter: int = 300
    tol: float = 1e-4
    algorithm: Literal["lloyd", "elkan", "auto", "full"] = "lloyd"
