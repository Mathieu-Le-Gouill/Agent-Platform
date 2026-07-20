from agent_platform.core.interfaces.clustering.base import BaseClusteringAlgorithm
from agent_platform.core.interfaces.clustering.config import ClusteringConfig
from agent_platform.integrations.clustering.gmm.config import GMMConfig
from agent_platform.integrations.clustering.hdbscan.config import HDBSCANConfig
from agent_platform.integrations.clustering.kmeans.config import KMeansConfig
from agent_platform.core.interfaces.clustering.response import (
    ClusterResult,
    ClusteredItem,
)


class TestClusteringInitExports:
    def test_clustering_algorithm_is_importable(self):
        assert BaseClusteringAlgorithm is not None

    def test_clustering_config_is_importable(self):
        assert ClusteringConfig is not None

    def test_hdbscan_config_is_importable(self):
        assert HDBSCANConfig is not None

    def test_kmeans_config_is_importable(self):
        assert KMeansConfig is not None

    def test_gmm_config_is_importable(self):
        assert GMMConfig is not None

    def test_cluster_result_is_importable(self):
        assert ClusterResult is not None

    def test_clustered_item_is_importable(self):
        assert ClusteredItem is not None
