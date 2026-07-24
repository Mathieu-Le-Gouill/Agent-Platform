from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.clustering.config import ClusteringConfig


class KMeansConfig(ClusteringConfig):
    n_clusters: int = 8  # number of clusters to form and centroids to generate
    init: str = "k-means++"  # method for centroid initialization ("k-means++", "random", or an array of initial centers)
    n_init: int | Literal["auto"] = (
        10  # number of times the algorithm runs with different centroid seeds; best result (by inertia) is kept
    )
    max_iter: int = 300  # maximum number of iterations for a single run
    tol: float = 1e-4  # relative tolerance on centroid movement to declare convergence
    algorithm: Literal["lloyd", "elkan"] = (
        "lloyd"  # K-means variant to use; "elkan" is faster on well-defined clusters via triangle inequality
    )
    random_state: int | None = (
        None  # seed controlling centroid initialization for reproducible results
    )
    copy_x: bool = True  # if True, the input data is copied before centering rather than modified in place
    verbose: int = 0  # verbosity level of the algorithm's logging


# sources: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
