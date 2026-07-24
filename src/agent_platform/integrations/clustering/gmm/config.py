from __future__ import annotations

from typing import Literal

from agent_platform.core.interfaces.clustering.config import ClusteringConfig


class GMMConfig(ClusteringConfig):
    # Note: the inherited `n_clusters` (base field) is not applicable to GMM — this provider uses `n_components` instead.
    n_components: int = 8  # number of mixture components (clusters) to fit
    covariance_type: Literal["full", "tied", "diag", "spherical"] = (
        "full"  # constraint on the shape of each component's covariance matrix
    )
    max_iter: int = 100  # maximum number of EM iterations to perform
    n_init: int = 1  # number of EM initializations to run; best result (by log-likelihood) is kept
    tol: float = 1e-3  # convergence threshold on the change in average log-likelihood
    # regularization added to the covariance diagonal, important for degenerate/low-variance embedding clusters
    reg_covar: float = 1e-6
    init_params: Literal["kmeans", "k-means++", "random", "random_from_data"] = (
        "kmeans"  # method used to initialize component weights, means, and covariances
    )


# sources: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
