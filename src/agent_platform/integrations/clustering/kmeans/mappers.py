from __future__ import annotations

import numpy as np

from agent_platform.core.interfaces.clustering.response import (
    ClusteredItem,
    ClusterResult,
)
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.cluster import Cluster
from agent_platform.core.schemas.document import TextDocument

__all__ = ["build_label", "build_result", "extract_vectors", "softmax"]


def extract_vectors(items: list[TextChunk]) -> list[list[float]]:
    return [item.metadata.get("embedding", []) for item in items]


def softmax(x: np.ndarray) -> np.ndarray:
    # Subtract row max before exponentiating to avoid overflow.
    e_x = np.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def build_label(label: int) -> str:
    return f"cluster_{label}"


def build_result(
    items: list[TextChunk],
    labels: list[int],
    probabilities: np.ndarray | None = None,
) -> ClusterResult:
    clusters_map: dict[int, list[TextChunk | TextDocument]] = {}
    for item, label in zip(items, labels):
        clusters_map.setdefault(label, []).append(item)

    clusters = [
        Cluster(label=build_label(cid), items=citems)
        for cid, citems in clusters_map.items()
    ]
    clustered_items = [
        ClusteredItem(
            index=i,
            cluster_id=label,
            label=build_label(label),
            probability=float(probabilities[i, label])
            if probabilities is not None
            else 0.0,
        )
        for i, label in enumerate(labels)
    ]
    return ClusterResult(clusters=clusters, items=clustered_items)
