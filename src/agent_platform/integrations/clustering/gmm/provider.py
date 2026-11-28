from __future__ import annotations

from sklearn.mixture import GaussianMixture

from agent_platform.core.interfaces.clustering.base import BaseClusteringAlgorithm
from agent_platform.core.interfaces.clustering.response import ClusterResult
from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.clustering.gmm.config import GMMConfig
from agent_platform.integrations.clustering.mappers import (
    build_result,
    extract_vectors,
    validate_vectors,
)


class GMMClusterer(BaseClusteringAlgorithm[GMMConfig]):
    def _default_config(self) -> GMMConfig:
        return GMMConfig()

    async def clusterize(
        self,
        items: list[TextChunk],
        config: GMMConfig | None = None,
    ) -> ClusterResult:
        config = config or self._default_config()
        if not items:
            return ClusterResult(clusters=[], items=[])
        validate_vectors(items)

        vectors = extract_vectors(items)
        n_components = config.n_components or min(8, len(items))
        n_components = min(n_components, len(items))

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=config.covariance_type,
            max_iter=config.max_iter,
            n_init=config.n_init,
            tol=config.tol,
            reg_covar=config.reg_covar,
            init_params=config.init_params,
            random_state=config.random_state,
        )
        labels = gmm.fit_predict(vectors)
        probabilities = gmm.predict_proba(vectors)

        return build_result(
            items,
            labels,
            [probabilities[i, label] for i, label in enumerate(labels)],
        )
