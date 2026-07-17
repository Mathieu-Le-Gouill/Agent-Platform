from __future__ import annotations
from typing import Literal
from agent_platform.core.interfaces.clustering.config import ClusteringConfig


class GMMConfig(ClusteringConfig):
    n_components: int = 8
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    max_iter: int = 100
    n_init: int = 1
    tol: float = 1e-3
