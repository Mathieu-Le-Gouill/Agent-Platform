from __future__ import annotations

from dataclasses import dataclass, field

from agent_platform.core.schemas.cluster import Cluster


@dataclass(slots=True, frozen=True)
class ClusteredItem:
    index: int
    cluster_id: int
    label: str
    probability: float


@dataclass(slots=True, frozen=True)
class ClusterResult:
    clusters: list[Cluster] = field(default_factory=list)
    items: list[ClusteredItem] = field(default_factory=list)
