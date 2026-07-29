from __future__ import annotations

from agent_platform.core.config import ProviderConfig


class ClusteringConfig(ProviderConfig):
    # seed for reproducible clustering; generic base field, not applicable to deterministic providers like HDBSCAN
    random_state: int | None = None
