import pytest

from agent_platform.core.interfaces.clustering.base import BaseClusteringAlgorithm
from agent_platform.core.interfaces.clustering.config import ClusteringConfig
from agent_platform.core.interfaces.clustering.response import ClusterResult
from agent_platform.core.credentials import ProviderCredentials


class _ConcreteClusteringAlgorithm(BaseClusteringAlgorithm[ClusteringConfig]):
    async def clusterize(self, items, config=None):
        return ClusterResult()


class _CredentialsStoringAlgorithm(BaseClusteringAlgorithm[ClusteringConfig]):
    def __init__(self, credentials: ProviderCredentials) -> None:
        self._credentials = credentials

    async def clusterize(self, items, config=None):
        return ClusterResult()


class TestClusteringAlgorithm:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            BaseClusteringAlgorithm()

    def test_concrete_subclass_can_be_instantiated(self):
        algo = _ConcreteClusteringAlgorithm()
        assert isinstance(algo, BaseClusteringAlgorithm)

    async def test_clusterize_abstract_method_can_be_overridden(self):
        algo = _ConcreteClusteringAlgorithm()
        result = await algo.clusterize(items=[])
        assert isinstance(result, ClusterResult)

    def test_stores_credentials_of_different_types(self):
        creds = ProviderCredentials(api_key="test-key")
        algo = _CredentialsStoringAlgorithm(credentials=creds)
        assert algo._credentials.api_key.get_secret_value() == "test-key"
