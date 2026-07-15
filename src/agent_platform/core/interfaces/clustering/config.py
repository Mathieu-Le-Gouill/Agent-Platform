from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ClusteringConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    n_clusters: int | None = None
    cluster_labels: list[str] | None = None
    random_state: int | None = None
