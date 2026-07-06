from agent_platform.integrations.clustering.config import (
    ClusteringConfig,
    HDBSCANConfig,
    KMeansConfig,
)


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


def test_kmeans_config_defaults():
    cfg = KMeansConfig()
    assert isinstance(cfg, ClusteringConfig)
    assert cfg.n_init == "auto"
    assert cfg.max_iter == 300
    assert cfg.tol == 1e-4
    assert cfg.algorithm == "lloyd"


def test_kmeans_config_construction():
    cfg = KMeansConfig(n_clusters=5, n_init=10, algorithm="elkan")
    assert cfg.n_clusters == 5
    assert cfg.n_init == 10
    assert cfg.algorithm == "elkan"


def test_configs_are_frozen():
    cfg = ClusteringConfig()
    try:
        cfg.n_clusters = 1
        assert False, "expected FrozenInstanceError"
    except AttributeError:
        pass

    hcfg = HDBSCANConfig()
    try:
        hcfg.min_cluster_size = 10
        assert False, "expected FrozenInstanceError"
    except AttributeError:
        pass

    kcfg = KMeansConfig()
    try:
        kcfg.max_iter = 100
        assert False, "expected FrozenInstanceError"
    except AttributeError:
        pass
