import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# hdbscan is not installed in this environment; mock at module level so the
# package __init__.py does not fail at import time.
sys.modules["hdbscan"] = MagicMock()

from agent_platform.integrations.clustering.config import KMeansConfig
from agent_platform.integrations.clustering.providers.hdbscan import (
    HDBSCANClusterer,
    _build_label_names as hdb_build_label_names,
)
from agent_platform.integrations.clustering.providers.kmeans import (
    KMeansClusterer,
    _softmax,
    _build_label_names as km_build_label_names,
)
from agent_platform.models.chunk import TextChunk


# ---------------------------------------------------------------------------
# Empty items
# ---------------------------------------------------------------------------


@patch("agent_platform.integrations.clustering.providers.hdbscan.hdbscan.HDBSCAN")
async def test_hdbscan_empty_items(mock_hdbscan):
    clusterer = HDBSCANClusterer()
    result = await clusterer.clusterize([])
    assert result.clusters == []
    assert result.items == []
    mock_hdbscan.assert_not_called()


@patch("agent_platform.integrations.clustering.providers.kmeans.KMeans")
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


@patch("agent_platform.integrations.clustering.providers.hdbscan.hdbscan.HDBSCAN")
async def test_hdbscan_clusterize(mock_hdbscan_cls):
    mock_instance = MagicMock()
    mock_instance.labels_ = np.array([0, 0, -1])
    mock_instance.probabilities_ = np.array([0.95, 0.92, 0.0])
    mock_instance.fit.return_value = mock_instance
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


@patch("agent_platform.integrations.clustering.providers.kmeans.KMeans")
async def test_kmeans_clusterize(mock_kmeans_cls):
    mock_instance = MagicMock()
    mock_instance.labels_ = np.array([0, 0, 1, 1])
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

    mock_instance.fit.assert_called_once()
    mock_instance.transform.assert_called_once()


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


# ---------------------------------------------------------------------------
# _build_label_names utility (kmeans flavour)
# ---------------------------------------------------------------------------


def test_build_label_names_none():
    labels = np.array([0, 0, 1, 1])
    assert km_build_label_names(labels, None) == {}


def test_build_label_names_empty_list():
    labels = np.array([0, 0, 1, 1])
    assert km_build_label_names(labels, []) == {}


def test_build_label_names_matching():
    labels = np.array([0, 0, 1, 1])
    assert km_build_label_names(labels, ["cat", "dog"]) == {0: "cat", 1: "dog"}


def test_build_label_names_fewer_user_labels():
    labels = np.array([0, 0, 1, 1, 2])
    result = km_build_label_names(labels, ["cat"])
    assert result == {0: "cat", 1: "cluster_1", 2: "cluster_2"}


def test_build_label_names_non_contiguous():
    labels = np.array([0, 0, 2, 3])
    result = km_build_label_names(labels, ["a", "b", "c"])
    assert result == {0: "a", 2: "b", 3: "c"}


# ---------------------------------------------------------------------------
# _build_label_names utility (hdbscan flavour — excludes -1)
# ---------------------------------------------------------------------------


def test_hdb_build_label_names_excludes_noise():
    labels = np.array([-1, 0, 0, 1, 1])
    result = hdb_build_label_names(labels, ["a", "b"])
    assert result == {0: "a", 1: "b"}


def test_hdb_build_label_names_none():
    labels = np.array([0, 0, 1, 1])
    assert hdb_build_label_names(labels, None) == {}
