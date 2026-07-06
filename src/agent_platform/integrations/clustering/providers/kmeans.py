from __future__ import annotations

import asyncio

import numpy as np
from sklearn.cluster import KMeans

from agent_platform.integrations.clustering.base import ClusteringAlgorithm
from agent_platform.integrations.clustering.config import KMeansConfig
from agent_platform.integrations.clustering.response import ClusterResult, ClusteredItem
from agent_platform.models.chunk import TextChunk
from agent_platform.models.cluster import Cluster


class KMeansClusterer(ClusteringAlgorithm[KMeansConfig]):
    def __init__(
        self,
        config: KMeansConfig | None = None,
    ) -> None:
        self._config = config or KMeansConfig()

    async def clusterize(
        self,
        items: list[TextChunk],
        config: KMeansConfig | None = None,
    ) -> ClusterResult:

        if not items:
            return ClusterResult()

        embeddings = np.array(
            [item.metadata.get("embedding", []) for item in items],
            dtype=np.float32,
        )

        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise ValueError(
                "Each TextChunk must have an 'embedding' metadata key "
                "with a numeric vector"
            )

        cfg = config or self._config

        n_clusters = cfg.n_clusters
        if n_clusters is None:
            raise ValueError(
                "KMeansClusterer requires n_clusters to be set. "
                "Provide it via KMeansConfig(n_clusters=...) or set it on ClusteringConfig."
            )

        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if n_clusters > len(items):
            raise ValueError("n_clusters cannot exceed the number of items")

        loop = asyncio.get_event_loop()

        clusterer = await loop.run_in_executor(
            None,
            lambda: KMeans(
                n_clusters=n_clusters,
                n_init=cfg.n_init,
                max_iter=cfg.max_iter,
                tol=cfg.tol,
                algorithm=cfg.algorithm,
                random_state=cfg.random_state,
            ).fit(embeddings),
        )

        labels: np.ndarray = clusterer.labels_
        distances: np.ndarray = clusterer.transform(embeddings)
        probabilities = _softmax(-distances)

        user_labels = cfg.cluster_labels
        label_names = _build_label_names(labels, user_labels)

        clusters: list[Cluster] = []
        items_out: list[ClusteredItem] = []

        unique_labels = sorted(set(labels))
        centroids = clusterer.cluster_centers_

        for label_id in unique_labels:
            mask = labels == label_id
            label_name = label_names.get(label_id, f"cluster_{label_id}")
            cluster_indices = list(np.where(mask)[0])
            cluster_items = [items[i] for i in cluster_indices]

            clusters.append(
                Cluster(
                    label=label_name,
                    items=cluster_items,
                    centroid=centroids[label_id].tolist(),
                )
            )

            for i in cluster_indices:
                items_out.append(
                    ClusteredItem(
                        index=i,
                        cluster_id=int(label_id),
                        label=label_name,
                        probability=float(probabilities[i, label_id]),
                    )
                )

        return ClusterResult(clusters=clusters, items=items_out)


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def _build_label_names(
    labels: np.ndarray,
    user_labels: list[str] | None,
) -> dict[int, str]:

    if not user_labels:
        return {}

    unique = sorted(set(labels))

    names: dict[int, str] = {}
    for i, label_id in enumerate(unique):
        if i < len(user_labels):
            names[label_id] = user_labels[i]
        else:
            names[label_id] = f"cluster_{label_id}"

    return names
