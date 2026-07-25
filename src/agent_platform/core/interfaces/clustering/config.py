from __future__ import annotations

from agent_platform.core.config import ProviderConfig


class ClusteringConfig(ProviderConfig):
    # target number of clusters, generic base field; GMM providers use `n_components` instead, HDBSCAN doesn't use this field at all
    n_clusters: int | None = None
    # optional human-readable labels assigned to each cluster index
    cluster_labels: list[str] | None = None
    # seed for reproducible clustering; generic base field, not applicable to deterministic providers like HDBSCAN
    random_state: int | None = None
