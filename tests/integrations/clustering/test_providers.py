import builtins
import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("sklearn")

from agent_platform.core.schemas.chunk import TextChunk
from agent_platform.integrations.clustering.gmm.config import GMMConfig
from agent_platform.integrations.clustering.gmm.gmm import GMMClusterer
from agent_platform.integrations.clustering.hdbscan.config import HDBSCANConfig
from agent_platform.integrations.clustering.hdbscan.hdbscan import (
    HDBSCANClusterer,
)
from agent_platform.integrations.clustering.kmeans.config import KMeansConfig
from agent_platform.integrations.clustering.kmeans.kmeans import (
    KMeansClusterer,
    _softmax,
)

# ---------------------------------------------------------------------------
# Empty items
# ---------------------------------------------------------------------------


@patch("hdbscan.HDBSCAN")
async def test_hdbscan_empty_items(mock_hdbscan):
    clusterer = HDBSCANClusterer()
    result = await clusterer.clusterize([])
    assert result.clusters == []
    assert result.items == []
    mock_hdbscan.assert_not_called()


@patch("agent_platform.integrations.clustering.kmeans.kmeans.KMeans")
async def test_kmeans_empty_items(mock_kmeans):
    clusterer = KMeansClusterer()
    result = await clusterer.clusterize([])
    assert result.clusters == []
    assert result.items == []
    mock_kmeans.assert_not_called()


# ---------------------------------------------------------------------------
# Missing embedding
# ---------------------------------------------------------------------------


async def test_hdbscan_missing_embedding_raises_value_error():
    clusterer = HDBSCANClusterer()
    item = TextChunk(text="hello", metadata={})
    with pytest.raises(ValueError, match="embedding"):
        await clusterer.clusterize([item])


async def test_kmeans_missing_embedding_raises_value_error():
    clusterer = KMeansClusterer()
    item = TextChunk(text="hello", metadata={})
    with pytest.raises(ValueError, match="embedding"):
        await clusterer.clusterize([item])


# ---------------------------------------------------------------------------
# KMeans n_clusters validation
# ---------------------------------------------------------------------------


async def test_kmeans_n_clusters_none_raises_value_error():
    clusterer = KMeansClusterer()
    item = TextChunk(text="a", metadata={"embedding": [0.1, 0.2]})
    with pytest.raises(ValueError, match="n_clusters"):
        await clusterer.clusterize([item])


async def test_kmeans_n_clusters_less_than_one_raises_value_error():
    clusterer = KMeansClusterer()
    items = [TextChunk(text="a", metadata={"embedding": [0.1, 0.2]})]
    cfg = KMeansConfig(n_clusters=0)
    with pytest.raises(ValueError, match=">= 1"):
        await clusterer.clusterize(items, config=cfg)


async def test_kmeans_n_clusters_exceeds_items():
    clusterer = KMeansClusterer()
    items = [
        TextChunk(text="a", metadata={"embedding": [0.1, 0.2]}),
        TextChunk(text="b", metadata={"embedding": [0.3, 0.4]}),
    ]
    cfg = KMeansConfig(n_clusters=5)
    with pytest.raises(ValueError, match="exceed"):
        await clusterer.clusterize(items, config=cfg)


# ---------------------------------------------------------------------------
# Full clustering flow
# ---------------------------------------------------------------------------


@patch("hdbscan.HDBSCAN")
async def test_hdbscan_clusterize(mock_hdbscan_cls):
    mock_instance = MagicMock()
    mock_instance.fit_predict.return_value = np.array([0, 0, -1])
    mock_instance.probabilities_ = np.array([0.95, 0.92, 0.0])
    mock_hdbscan_cls.return_value = mock_instance

    clusterer = HDBSCANClusterer()
    items = [
        TextChunk(text="a", metadata={"embedding": [0.1, 0.2]}),
        TextChunk(text="b", metadata={"embedding": [0.1, 0.2]}),
        TextChunk(text="c", metadata={"embedding": [0.5, 0.6]}),
    ]
    result = await clusterer.clusterize(items)

    assert len(result.clusters) == 2
    assert len(result.items) == 3

    cluster_0 = result.clusters[0]
    noise = result.clusters[1]
    assert cluster_0.label == "cluster_0"
    assert noise.label == "noise"

    assert result.items[0].cluster_id == 0
    assert result.items[0].label == "cluster_0"
    assert result.items[0].probability == 0.95
    assert result.items[2].cluster_id == -1
    assert result.items[2].label == "noise"
    assert result.items[2].probability == 0.0


@patch("agent_platform.integrations.clustering.kmeans.kmeans.KMeans")
async def test_kmeans_clusterize(mock_kmeans_cls):
    mock_instance = MagicMock()
    mock_instance.fit_predict.return_value = np.array([0, 0, 1, 1])
    mock_instance.cluster_centers_ = np.array([[0.1, 0.2], [0.3, 0.4]])
    mock_instance.transform.return_value = np.array(
        [
            [0.5, 3.0],
            [0.6, 2.9],
            [3.0, 0.5],
            [2.9, 0.6],
        ]
    )
    mock_instance.fit.return_value = mock_instance
    mock_kmeans_cls.return_value = mock_instance

    clusterer = KMeansClusterer()
    items = [
        TextChunk(text="a", metadata={"embedding": [0.1, 0.2]}),
        TextChunk(text="b", metadata={"embedding": [0.1, 0.2]}),
        TextChunk(text="c", metadata={"embedding": [0.3, 0.4]}),
        TextChunk(text="d", metadata={"embedding": [0.3, 0.4]}),
    ]
    cfg = KMeansConfig(n_clusters=2, random_state=42)
    result = await clusterer.clusterize(items, config=cfg)

    assert len(result.clusters) == 2
    assert len(result.items) == 4

    assert result.items[0].cluster_id == 0
    assert result.items[0].label == "cluster_0"
    assert result.items[2].cluster_id == 1
    assert result.items[2].label == "cluster_1"

    mock_instance.fit_predict.assert_called_once()
    mock_instance.transform.assert_called_once()


# ---------------------------------------------------------------------------
# KMeans config forwarding
# ---------------------------------------------------------------------------


@patch("agent_platform.integrations.clustering.kmeans.kmeans.KMeans")
async def test_kmeans_forwards_init_to_constructor(mock_kmeans_cls):
    mock_instance = MagicMock()
    mock_instance.fit_predict.return_value = np.array([0])
    mock_instance.transform.return_value = np.array([[0.1]])
    mock_kmeans_cls.return_value = mock_instance

    clusterer = KMeansClusterer()
    items = [TextChunk(text="a", metadata={"embedding": [0.1, 0.2]})]
    cfg = KMeansConfig(n_clusters=1, init="random")
    await clusterer.clusterize(items, config=cfg)

    _, kwargs = mock_kmeans_cls.call_args
    assert kwargs["init"] == "random"


@patch("agent_platform.integrations.clustering.kmeans.kmeans.KMeans")
async def test_kmeans_forwards_copy_x_and_verbose(mock_kmeans_cls):
    mock_instance = MagicMock()
    mock_instance.fit_predict.return_value = np.array([0])
    mock_instance.transform.return_value = np.array([[0.1]])
    mock_kmeans_cls.return_value = mock_instance

    clusterer = KMeansClusterer()
    items = [TextChunk(text="a", metadata={"embedding": [0.1, 0.2]})]
    cfg = KMeansConfig(n_clusters=1, copy_x=False, verbose=5)
    await clusterer.clusterize(items, config=cfg)

    _, kwargs = mock_kmeans_cls.call_args
    assert kwargs["copy_x"] is False
    assert kwargs["verbose"] == 5


# ---------------------------------------------------------------------------
# GMM clusterize + config forwarding
# ---------------------------------------------------------------------------


@patch("agent_platform.integrations.clustering.gmm.gmm.GaussianMixture")
async def test_gmm_clusterize(mock_gmm_cls):
    mock_instance = MagicMock()
    mock_instance.fit_predict.return_value = np.array([0, 0, 1, 1])
    mock_instance.predict_proba.return_value = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
    )
    mock_gmm_cls.return_value = mock_instance

    clusterer = GMMClusterer()
    items = [
        TextChunk(text="a", metadata={"embedding": [0.1, 0.2]}),
        TextChunk(text="b", metadata={"embedding": [0.1, 0.2]}),
        TextChunk(text="c", metadata={"embedding": [0.3, 0.4]}),
        TextChunk(text="d", metadata={"embedding": [0.3, 0.4]}),
    ]
    cfg = GMMConfig(n_components=2)
    result = await clusterer.clusterize(items, config=cfg)

    assert len(result.clusters) == 2
    assert len(result.items) == 4
    assert result.items[0].cluster_id == 0
    assert result.items[2].cluster_id == 1


@patch("agent_platform.integrations.clustering.gmm.gmm.GaussianMixture")
async def test_gmm_forwards_reg_covar_and_init_params(mock_gmm_cls):
    mock_instance = MagicMock()
    mock_instance.fit_predict.return_value = np.array([0])
    mock_instance.predict_proba.return_value = np.array([[1.0]])
    mock_gmm_cls.return_value = mock_instance

    clusterer = GMMClusterer()
    items = [TextChunk(text="a", metadata={"embedding": [0.1, 0.2]})]
    cfg = GMMConfig(n_components=1, reg_covar=1e-3, init_params="random")
    await clusterer.clusterize(items, config=cfg)

    _, kwargs = mock_gmm_cls.call_args
    assert kwargs["reg_covar"] == 1e-3
    assert kwargs["init_params"] == "random"


# ---------------------------------------------------------------------------
# HDBSCAN config forwarding
# ---------------------------------------------------------------------------


@patch("hdbscan.HDBSCAN")
async def test_hdbscan_forwards_new_fields(mock_hdbscan_cls):
    mock_instance = MagicMock()
    mock_instance.fit_predict.return_value = np.array([0])
    mock_instance.probabilities_ = np.array([0.9])
    mock_hdbscan_cls.return_value = mock_instance

    clusterer = HDBSCANClusterer()
    items = [TextChunk(text="a", metadata={"embedding": [0.1, 0.2]})]
    cfg = HDBSCANConfig(
        alpha=0.7,
        allow_single_cluster=True,
        cluster_selection_epsilon_max=3.0,
        max_cluster_size=50,
        cluster_selection_method="leaf",
    )
    await clusterer.clusterize(items, config=cfg)

    _, kwargs = mock_hdbscan_cls.call_args
    assert kwargs["alpha"] == 0.7
    assert kwargs["allow_single_cluster"] is True
    assert kwargs["cluster_selection_epsilon_max"] == 3.0
    assert kwargs["max_cluster_size"] == 50
    assert kwargs["cluster_selection_method"] == "leaf"


def test_hdbscan_module_import_error_surfaces_at_load_time(monkeypatch):
    monkeypatch.delitem(sys.modules, "hdbscan", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "agent_platform.integrations.clustering.hdbscan.hdbscan",
        raising=False,
    )

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "hdbscan":
            raise ImportError("simulated missing hdbscan")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        importlib.import_module(
            "agent_platform.integrations.clustering.hdbscan.hdbscan"
        )


# ---------------------------------------------------------------------------
# _softmax utility
# ---------------------------------------------------------------------------


def test_softmax_shape_and_sum():
    x = np.array([[1.0, 2.0, 3.0]])
    result = _softmax(x)
    assert result.shape == (1, 3)
    np.testing.assert_allclose(result.sum(axis=1), [1.0], rtol=1e-6)


def test_softmax_monotonic():
    x = np.array([[1.0, 2.0, 3.0]])
    result = _softmax(x)
    assert result[0, 0] < result[0, 1] < result[0, 2]


def test_softmax_multi_row():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _softmax(x)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], rtol=1e-6)


def test_softmax_against_definition():
    x = np.array([[1.0, 2.0, 3.0]])
    e_x = np.exp(x - x.max(axis=1, keepdims=True))
    expected = e_x / e_x.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(_softmax(x), expected, rtol=1e-6)
