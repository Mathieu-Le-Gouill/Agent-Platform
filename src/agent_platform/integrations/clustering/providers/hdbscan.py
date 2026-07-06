from __future__ import annotations

import asyncio

import hdbscan
import numpy as np

from agent_platform.integrations.clustering.base import ClusteringAlgorithm
from agent_platform.integrations.clustering.config import HDBSCANConfig
from agent_platform.integrations.clustering.response import ClusterResult, ClusteredItem
from agent_platform.models.chunk import TextChunk
from agent_platform.models.cluster import Cluster


class HDBSCANClusterer(ClusteringAlgorithm[HDBSCANConfig]):
    def __init__(
        self,
        config: HDBSCANConfig | None = None,
    ) -> None:
        self._config = config or HDBSCANConfig()

    async def clusterize(
        self,
        items: list[TextChunk],
        config: HDBSCANConfig | None = None,
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

        loop = asyncio.get_event_loop()

        clusterer = await loop.run_in_executor(
            None,
            lambda: hdbscan.HDBSCAN(
                min_cluster_size=cfg.min_cluster_size,
                min_samples=cfg.min_samples,
                metric=cfg.metric,
                cluster_selection_epsilon=cfg.cluster_selection_epsilon,
            ).fit(embeddings),
        )

        labels: np.ndarray = clusterer.labels_
        probabilities: np.ndarray = clusterer.probabilities_

        user_labels = cfg.cluster_labels
        label_names = _build_label_names(labels, user_labels)

        clusters: list[Cluster] = []
        items_out: list[ClusteredItem] = []

        unique_labels = set(labels)
        has_noise = -1 in unique_labels
        if has_noise:
            unique_labels.discard(-1)

        for label_id in sorted(unique_labels):
            mask = labels == label_id
            label_name = label_names.get(label_id, f"cluster_{label_id}")
            cluster_indices = list(np.where(mask)[0])
            cluster_items = [items[i] for i in cluster_indices]
            centroid = embeddings[mask].mean(axis=0).tolist() if mask.any() else None

            clusters.append(
                Cluster(
                    label=label_name,
                    items=cluster_items,
                    centroid=centroid,
                )
            )

            for i in cluster_indices:
                items_out.append(
                    ClusteredItem(
                        index=i,
                        cluster_id=int(label_id),
                        label=label_name,
                        probability=float(probabilities[i]),
                    )
                )

        if has_noise:
            noise_mask = labels == -1
            noise_indices = list(np.where(noise_mask)[0])
            noise_items = [items[i] for i in noise_indices]

            clusters.append(
                Cluster(
                    label="noise",
                    items=noise_items,
                )
            )

            for i in noise_indices:
                items_out.append(
                    ClusteredItem(
                        index=i,
                        cluster_id=-1,
                        label="noise",
                        probability=0.0,
                    )
                )

        return ClusterResult(clusters=clusters, items=items_out)


def _build_label_names(
    labels: np.ndarray,
    user_labels: list[str] | None,
) -> dict[int, str]:

    if not user_labels:
        return {}

    unique = sorted(set(l for l in labels if l != -1))

    names: dict[int, str] = {}
    for i, label_id in enumerate(unique):
        if i < len(user_labels):
            names[label_id] = user_labels[i]
        else:
            names[label_id] = f"cluster_{label_id}"

    return names
