from uuid import uuid4

from agent_platform.models.cluster import Cluster
from agent_platform.integrations.clustering.response import ClusterResult, ClusteredItem


def test_clustered_item_construction():
    item = ClusteredItem(index=0, cluster_id=1, label="test", probability=0.95)
    assert item.index == 0
    assert item.cluster_id == 1
    assert item.label == "test"
    assert item.probability == 0.95


def test_clustered_item_is_frozen():
    item = ClusteredItem(index=0, cluster_id=1, label="test", probability=0.95)
    try:
        item.label = "changed"
        assert False, "expected FrozenInstanceError"
    except AttributeError:
        pass


def test_cluster_result_defaults():
    result = ClusterResult()
    assert result.clusters == []
    assert result.items == []


def test_cluster_result_with_items():
    cid = uuid4()
    cluster = Cluster(label="test", id=cid)
    clustered_item = ClusteredItem(index=0, cluster_id=0, label="test", probability=1.0)
    result = ClusterResult(clusters=[cluster], items=[clustered_item])
    assert len(result.clusters) == 1
    assert result.clusters[0].id == cid
    assert len(result.items) == 1
    assert result.items[0] == clustered_item


def test_cluster_result_is_frozen():
    result = ClusterResult()
    try:
        result.clusters = [Cluster(label="x")]
        assert False, "expected FrozenInstanceError"
    except AttributeError:
        pass
