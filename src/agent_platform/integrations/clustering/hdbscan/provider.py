from __future__ import annotations

import hdbscan

from agent_platform.core.interfaces.clustering.base import BaseClusteringAlgorithm
from agent_platform.core.interfaces.clustering.response import ClusterResult
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.clustering.hdbscan.config import HDBSCANConfig
from agent_platform.integrations.clustering.hdbscan.mappers import (
    build_result,
    extract_vectors,
    validate_vectors,
)


class HDBSCANClusterer(BaseClusteringAlgorithm[HDBSCANConfig]):
    def _default_config(self) -> HDBSCANConfig:
        return HDBSCANConfig()

    async def clusterize(
        self,
        items: list[TextChunk],
        config: HDBSCANConfig | None = None,
    ) -> ClusterResult:
        config = config or self._default_config()
        if not items:
            return ClusterResult(clusters=[], items=[])
        validate_vectors(items)

        vectors = extract_vectors(items)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            metric=config.metric,
            cluster_selection_epsilon=config.cluster_selection_epsilon,
            cluster_selection_method=config.cluster_selection_method,
            alpha=config.alpha,
            allow_single_cluster=config.allow_single_cluster,
            cluster_selection_epsilon_max=config.cluster_selection_epsilon_max,
            max_cluster_size=config.max_cluster_size,
        )
        labels = clusterer.fit_predict(vectors)
        probabilities = getattr(clusterer, "probabilities_", None)

        return build_result(items, labels, probabilities)
