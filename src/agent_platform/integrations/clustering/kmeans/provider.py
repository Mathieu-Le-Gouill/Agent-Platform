from __future__ import annotations

from sklearn.cluster import KMeans

from agent_platform.core.interfaces.clustering.base import BaseClusteringAlgorithm
from agent_platform.core.interfaces.clustering.response import ClusterResult
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.clustering.kmeans.config import KMeansConfig
from agent_platform.integrations.clustering.kmeans.mappers import (
    build_result,
    extract_vectors,
    softmax,
)


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

        vectors = extract_vectors(items)
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
        probs = softmax(-distances)

        return build_result(items, labels, probs)
