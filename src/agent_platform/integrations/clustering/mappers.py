"""Conversion shared by every clustering provider: embeddings in, result out."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agent_platform.core.interfaces.clustering.response import (
    ClusteredItem,
    ClusterResult,
)
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.core.schemas.cluster import Cluster
from agent_platform.core.schemas.document import TextDocument

__all__ = ["build_result", "cluster_label", "extract_vectors", "validate_vectors"]


def validate_vectors(items: list[TextChunk]) -> None:
    for item in items:
        if "embedding" not in item.metadata:
            raise ValueError("Item missing embedding in metadata")


def extract_vectors(items: list[TextChunk]) -> list[list[float]]:
    return [item.metadata["embedding"] for item in items]


def cluster_label(label: int) -> str:
    return f"cluster_{label}"


def build_result(
    items: list[TextChunk],
    labels: Sequence[int],
    probabilities: Sequence[float] | None = None,
    label_name: Callable[[int], str] = cluster_label,
) -> ClusterResult:
    """`probabilities[i]` is item i's membership probability in its own cluster."""
    cluster_ids = [int(label) for label in labels]
    clusters_map: dict[int, list[TextChunk | TextDocument]] = {}
    for item, cluster_id in zip(items, cluster_ids):
        clusters_map.setdefault(cluster_id, []).append(item)

    clusters = [
        Cluster(label=label_name(cid), items=citems)
        for cid, citems in clusters_map.items()
    ]
    clustered_items = [
        ClusteredItem(
            index=i,
            cluster_id=cluster_id,
            label=label_name(cluster_id),
            probability=float(probabilities[i]) if probabilities is not None else 0.0,
        )
        for i, cluster_id in enumerate(cluster_ids)
    ]
    return ClusterResult(clusters=clusters, items=clustered_items)
