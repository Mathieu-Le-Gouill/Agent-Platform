from __future__ import annotations

import hdbscan

from agent_platform.core.interfaces.clustering.base import BaseClusteringAlgorithm
from agent_platform.integrations.clustering.hdbscan.config import HDBSCANConfig
from agent_platform.core.interfaces.clustering.response import (
    ClusterResult,
    ClusteredItem,
)
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.cluster import Cluster


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
        _validate_vectors(items)

        vectors = _extract_vectors(items)
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

        return _build_result(items, labels, probabilities)


def _validate_vectors(items: list[TextChunk]) -> None:
    for item in items:
        if "embedding" not in item.metadata:
            raise ValueError("Item missing embedding in metadata")


def _extract_vectors(items: list[TextChunk]) -> list[list[float]]:
    return [item.metadata.get("embedding", []) for item in items]


def _build_label(label: int) -> str:
    return f"cluster_{label}" if label >= 0 else "noise"


def _build_result(
    items: list[TextChunk],
    labels: list[int],
    probabilities: list[float] | None = None,
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
            probability=float(probabilities[i]) if probabilities is not None else 0.0,
        )
        for i, label in enumerate(labels)
    ]
    return ClusterResult(clusters=clusters, items=clustered_items)
