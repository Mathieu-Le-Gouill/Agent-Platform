from __future__ import annotations

from agent_platform.core.interfaces.clustering.config import ClusteringConfig


class HDBSCANConfig(ClusteringConfig):
    # Note: the inherited `random_state` (base field) is not applicable to HDBSCAN — the algorithm is deterministic given the same data/params.
    min_cluster_size: int = 5  # smallest grouping of points considered a cluster
    min_samples: int | None = (
        None  # neighborhood size used to estimate density; higher values yield more conservative clustering, defaults to min_cluster_size
    )
    cluster_selection_epsilon: float = 0.0  # distance threshold below which micro-clusters are merged, avoids over-splitting close clusters
    metric: str = (
        "euclidean"  # distance metric used to compute pairwise distances between points
    )
    cluster_selection_method: str = "eom"  # how flat clusters are extracted from the condensed tree: "eom" (excess of mass) or "leaf"
    # distance scaling factor, affects how conservative clustering is
    alpha: float = 1.0
    allow_single_cluster: bool = (
        False  # if True, permits the entire dataset to be returned as a single cluster
    )
    cluster_selection_epsilon_max: float = float(
        "inf"
    )  # distance threshold above which cluster splits are no longer merged back together
    max_cluster_size: int = 0  # upper bound on cluster size; 0 means no limit


"""
sources: https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
         https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.HDBSCAN
"""
