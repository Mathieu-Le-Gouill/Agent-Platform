from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from agent_platform.core.interfaces.clustering.base import BaseClusteringAlgorithm
from agent_platform.integrations.clustering.gmm.config import GMMConfig
from agent_platform.core.interfaces.clustering.response import (
    ClusterResult,
    ClusteredItem,
)
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.cluster import Cluster


class GMMClusterer(BaseClusteringAlgorithm[GMMConfig]):
    def _default_config(self) -> GMMConfig:
        return GMMConfig()

    async def clusterize(
        self,
        items: list[TextChunk],
        config: GMMConfig | None = None,
    ) -> ClusterResult:
        config = config or self._default_config()
        if not items:
            return ClusterResult(clusters=[], items=[])

        for item in items:
            if "embedding" not in item.metadata:
                raise ValueError("Item missing embedding in metadata")

        vectors = np.array([item.metadata["embedding"] for item in items])
        n_components = config.n_components or min(8, len(items))
        n_components = min(n_components, len(items))

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=config.covariance_type,
            max_iter=config.max_iter,
            n_init=config.n_init,
            tol=config.tol,
            reg_covar=config.reg_covar,
            init_params=config.init_params,
            random_state=config.random_state,
        )
        labels = gmm.fit_predict(vectors)
        probabilities = gmm.predict_proba(vectors)

        clusters_map: dict[int, list[TextChunk]] = {}
        for item, label in zip(items, labels):
            clusters_map.setdefault(int(label), []).append(item)

        clusters = [
            Cluster(label=f"cluster_{cid}", items=citems)
            for cid, citems in clusters_map.items()
        ]
        clustered_items = [
            ClusteredItem(
                index=i,
                cluster_id=int(label),
                label=f"cluster_{label}",
                probability=float(probabilities[i, int(label)]),
            )
            for i, label in enumerate(labels)
        ]

        return ClusterResult(clusters=clusters, items=clustered_items)
