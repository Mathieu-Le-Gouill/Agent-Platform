from pydantic import ValidationError

from agent_platform.core.interfaces.clustering.config import ClusteringConfig
from agent_platform.integrations.clustering.gmm.config import GMMConfig
from agent_platform.integrations.clustering.hdbscan.config import HDBSCANConfig
from agent_platform.integrations.clustering.kmeans.config import KMeansConfig


def test_clustering_config_defaults():
    cfg = ClusteringConfig()
    assert cfg.n_clusters is None
    assert cfg.cluster_labels is None
    assert cfg.random_state is None


def test_clustering_config_construction():
    cfg = ClusteringConfig(n_clusters=3, cluster_labels=["a", "b"], random_state=42)
    assert cfg.n_clusters == 3
    assert cfg.cluster_labels == ["a", "b"]
    assert cfg.random_state == 42


def test_hdbscan_config_defaults():
    cfg = HDBSCANConfig()
    assert isinstance(cfg, ClusteringConfig)
    assert cfg.min_cluster_size == 5
    assert cfg.min_samples is None
    assert cfg.metric == "euclidean"
    assert cfg.cluster_selection_epsilon == 0.0


def test_hdbscan_config_construction():
    cfg = HDBSCANConfig(min_cluster_size=10, metric="cosine")
    assert cfg.min_cluster_size == 10
    assert cfg.metric == "cosine"
    assert cfg.min_samples is None


def test_hdbscan_config_new_fields_defaults():
    cfg = HDBSCANConfig()
    assert cfg.alpha == 1.0
    assert cfg.allow_single_cluster is False
    assert cfg.cluster_selection_epsilon_max == float("inf")
    assert cfg.max_cluster_size == 0


def test_hdbscan_config_new_fields_construction():
    cfg = HDBSCANConfig(
        alpha=0.5,
        allow_single_cluster=True,
        cluster_selection_epsilon_max=2.0,
        max_cluster_size=100,
    )
    assert cfg.alpha == 0.5
    assert cfg.allow_single_cluster is True
    assert cfg.cluster_selection_epsilon_max == 2.0
    assert cfg.max_cluster_size == 100


def test_kmeans_config_defaults():
    cfg = KMeansConfig()
    assert isinstance(cfg, ClusteringConfig)
    assert cfg.n_init == 10
    assert cfg.max_iter == 300
    assert cfg.tol == 1e-4
    assert cfg.algorithm == "lloyd"
    assert cfg.copy_x is True
    assert cfg.verbose == 0


def test_kmeans_config_construction():
    cfg = KMeansConfig(n_clusters=5, n_init=10, algorithm="elkan")
    assert cfg.n_clusters == 5
    assert cfg.n_init == 10
    assert cfg.algorithm == "elkan"


def test_kmeans_config_new_fields_construction():
    cfg = KMeansConfig(copy_x=False, verbose=2)
    assert cfg.copy_x is False
    assert cfg.verbose == 2


def test_gmm_config_defaults():
    cfg = GMMConfig()
    assert isinstance(cfg, ClusteringConfig)
    assert cfg.n_components == 8
    assert cfg.covariance_type == "full"
    assert cfg.max_iter == 100
    assert cfg.n_init == 1
    assert cfg.tol == 1e-3
    assert cfg.reg_covar == 1e-6
    assert cfg.init_params == "kmeans"


def test_gmm_config_construction():
    cfg = GMMConfig(n_components=4, reg_covar=1e-4, init_params="random")
    assert cfg.n_components == 4
    assert cfg.reg_covar == 1e-4
    assert cfg.init_params == "random"


def test_configs_are_frozen():
    cfg = ClusteringConfig()
    try:
        cfg.n_clusters = 1
        assert False, "expected ValidationError"
    except ValidationError:
        pass

    hcfg = HDBSCANConfig()
    try:
        hcfg.min_cluster_size = 10
        assert False, "expected ValidationError"
    except ValidationError:
        pass

    kcfg = KMeansConfig()
    try:
        kcfg.max_iter = 100
        assert False, "expected ValidationError"
    except ValidationError:
        pass
