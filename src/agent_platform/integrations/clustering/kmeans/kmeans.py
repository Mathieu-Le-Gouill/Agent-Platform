from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from agent_platform.core.interfaces.clustering.base import BaseClusteringAlgorithm
from agent_platform.integrations.clustering.kmeans.config import KMeansConfig
from agent_platform.core.interfaces.clustering.response import (
    ClusterResult,
    ClusteredItem,
)
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.cluster import Cluster


class KMeansClusterer(BaseClusteringAlgorithm[KMeansConfig]):
    def _default_config(self) -> KMeansConfig:
        return KMeansConfig()

    async def clusterize(
        self,
        items: list[TextChunk],
        config: KMeansConfig | None = None,
    ) -> ClusterResult:
        config = config or self._default_config()

        if not items:
            return ClusterResult(clusters=[], items=[])
        for item in items:
            if "embedding" not in item.metadata:
                raise ValueError("Item missing embedding in metadata")

        vectors = _extract_vectors(items)
        if config.n_clusters is None:
            raise ValueError("n_clusters must be provided")
        if config.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if config.n_clusters > len(items):
            raise ValueError(
                f"n_clusters ({config.n_clusters}) cannot exceed number of items ({len(items)})"
            )
        n = config.n_clusters or min(8, len(items))

        clusterer = KMeans(
            n_clusters=n,
            init=config.init,
            n_init=config.n_init,
            max_iter=config.max_iter,
            tol=config.tol,
            algorithm=config.algorithm,
            random_state=config.random_state,
            copy_x=config.copy_x,
            verbose=config.verbose,
        )
        labels = clusterer.fit_predict(vectors)
        distances = clusterer.transform(vectors)
        probs = _softmax(-distances)

        return _build_result(items, labels, probs)


def _extract_vectors(items: list[TextChunk]) -> list[list[float]]:
    return [item.metadata.get("embedding", []) for item in items]


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def _build_label(label: int) -> str:
    return f"cluster_{label}"


def _build_result(
    items: list[TextChunk],
    labels: list[int],
    probabilities: np.ndarray | None = None,
) -> ClusterResult:
    clusters_map: dict[int, list[TextChunk]] = {}
    for item, label in zip(items, labels):
        clusters_map.setdefault(label, []).append(item)

    clusters = [
        Cluster(label=_build_label(cid), items=citems)
        for cid, citems in clusters_map.items()
    ]
    clustered_items = [
        ClusteredItem(
            index=i,
            cluster_id=label,
            label=_build_label(label),
            probability=float(probabilities[i, label])
            if probabilities is not None
            else 0.0,
        )
        for i, label in enumerate(labels)
    ]
    return ClusterResult(clusters=clusters, items=clustered_items)
